/**
 * 用 DevTools 协议驱动 Edge，把 HTML 打成 PDF。
 *
 * ## 为什么不用 `msedge --headless --print-to-pdf`
 *
 * 那个命令行开关**没法自定义页眉页脚**，只有两个选择：
 *
 *   - 不加 `--no-pdf-header-footer`：得到 Chromium 的默认页脚，
 *     左边印着 `file:///D:/Learning/Pytorch/MNIST_CNN/_book_merged.html`、
 *     右边印页码。一本书的页脚上印着本地磁盘路径，很荒唐。
 *   - 加了：页码也没了。全书两百多页，没有页码没法用。
 *
 * 想要「只有页码、居中、灰色小字」这一个组合，只能走 `Page.printToPDF`
 * 的 `footerTemplate`，而它**只在 CDP 上暴露**，命令行够不着。
 *
 * ## 为什么不用 puppeteer
 *
 * Node 22 起自带全局 `WebSocket`，裸连 CDP 只要几十行。为了一个页脚
 * 往仓库里塞几百 MB 的 node_modules 不划算。
 *
 * ## 用法
 *
 *     node tests/print_pdf.js <html> <pdf> [截图目录] [定位...]
 *
 * 给了截图目录就顺带截几张 PNG —— 用来核对排版（把 media 模拟成 print，
 * 否则截出来的是屏幕版，看不到打印时才生效的折行）。
 * 环境变量 `EDGE_PATH` 可以指定别的 Chromium 内核浏览器。
 *
 * 「定位」有两种写法，随便混用：
 *
 *     y:8000        滚到纵坐标 8000
 *     sel:blockquote.warn   滚到第一个匹配的元素
 *
 * 用选择器比算像素靠谱得多：正文里目录长短一变，像素偏移就全错了，
 * 而「滚到第一个警告框」永远指的是同一个东西。
 */
'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const EDGE = process.env.EDGE_PATH ||
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

/**
 * 页脚模板。pageNumber / totalPages 是 Chromium 认的特殊 class 名，
 * 会被替换成当前页码和总页数。
 *
 * 环境变量 `PDF_FOOTER` 可以覆盖它，传空字符串就是不要页脚
 * （比如要拿去再排版的稿子）。
 */
const FOOTER = process.env.PDF_FOOTER !== undefined
  ? process.env.PDF_FOOTER
  : [
    '<div style="width:100%;font-size:9px;color:#6a737d;text-align:center;',
    'font-family:\'Microsoft YaHei\',sans-serif;">',
    '<span class="pageNumber"></span> / <span class="totalPages"></span>',
    '</div>',
  ].join('');

/** 极简 CDP 客户端：只做「发一条命令、等一条回复」。 */
class CDP {
  constructor(ws) {
    this.ws = ws;
    this.seq = 0;
    this.pending = new Map();
    this.handlers = new Map();
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id !== undefined) {
        const slot = this.pending.get(msg.id);
        if (!slot) return;
        this.pending.delete(msg.id);
        if (msg.error) slot.reject(new Error(`${msg.error.message} (${msg.error.code})`));
        else slot.resolve(msg.result);
      } else {
        const fn = this.handlers.get(msg.method);
        if (fn) fn(msg.params);
      }
    });
  }

  send(method, params = {}, sessionId) {
    const id = ++this.seq;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    this.ws.send(JSON.stringify(payload));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
  }

  once(method) {
    return new Promise((resolve) => this.handlers.set(method, resolve));
  }
}

/**
 * 起 Edge，等它把 DevTools 的 ws 地址打到 stderr 上。
 *
 * 用 `--remote-debugging-port=0` 让系统随机分配端口，再把实际端口从
 * stderr 里读出来 —— 写死端口的话，同时开两个构建就会互相抢。
 */
function launchEdge() {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'edge-pdf-'));
  const proc = spawn(EDGE, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    // 页面是 file:// 的，图片和 CSS 也是 file:// ，
    // 不加这个 Chromium 可能拒绝加载同源的本地子资源
    '--allow-file-access-from-files',
    `--user-data-dir=${profile}`,
    '--remote-debugging-port=0',
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });

  const wsUrl = new Promise((resolve, reject) => {
    let buf = '';
    const timer = setTimeout(
      () => reject(new Error(`等 DevTools 端口超时。Edge 输出：\n${buf}`)), 30000);
    proc.stderr.on('data', (chunk) => {
      buf += chunk.toString();
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) { clearTimeout(timer); resolve(m[1]); }
    });
    proc.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`Edge 提前退出（code=${code}）。输出：\n${buf}`));
    });
  });

  return { proc, profile, wsUrl };
}

/** 打开 ws，等连上。 */
function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.addEventListener('open', () => resolve(ws));
    ws.addEventListener('error', () => reject(new Error(`连不上 DevTools: ${wsUrl}`)));
  });
}

async function main() {
  const [htmlArg, pdfArg, shotDir, ...scrollArgs] = process.argv.slice(2);
  if (!htmlArg || !pdfArg) {
    console.error('用法: node tests/print_pdf.js <html> <pdf> [截图目录] [scrollY ...]');
    return 2;
  }
  const htmlPath = path.resolve(htmlArg);
  const pdfPath = path.resolve(pdfArg);
  if (!fs.existsSync(htmlPath)) {
    console.error(`找不到 HTML: ${htmlPath}`);
    return 2;
  }

  const { proc, profile, wsUrl } = launchEdge();
  let ws;
  try {
    ws = await connect(await wsUrl);
    const cdp = new CDP(ws);

    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, sessionId);

    const loaded = cdp.once('Page.loadEventFired');
    await cdp.send('Page.navigate', { url: pathToFileURL(htmlPath).href }, sessionId);
    await loaded;

    // load 事件不代表图片解码完了。不等的话，PDF 里会出现一片空白图位。
    await cdp.send('Runtime.evaluate', {
      expression: `
        Promise.all(
          [...document.images].map((img) =>
            img.complete ? Promise.resolve() : new Promise((r) => {
              img.addEventListener('load', r);
              img.addEventListener('error', r);
            })
          )
        ).then(() => document.fonts.ready)
      `,
      awaitPromise: true,
    }, sessionId);

    // 渲染体检：数字对不上就不用往下看了，说明拼接或路径出了问题
    const stats = JSON.parse((await cdp.send('Runtime.evaluate', {
      expression: `JSON.stringify({
        height: document.body.scrollHeight,
        images: document.images.length,
        broken: [...document.images].filter((i) => !i.naturalWidth).length,
        h1: document.querySelectorAll('h1').length,
        h2: document.querySelectorAll('h2').length,
        pre: document.querySelectorAll('pre').length,
        warn: document.querySelectorAll('blockquote.warn').length,
        tip: document.querySelectorAll('blockquote.tip').length,
        done: document.querySelectorAll('blockquote.done').length,
        table: document.querySelectorAll('table').length,
      })`,
      returnByValue: true,
    }, sessionId)).result.value);
    console.log(`[页面] 高 ${stats.height}px ｜ h1 ${stats.h1} h2 ${stats.h2} ｜ `
      + `代码块 ${stats.pre} 表格 ${stats.table}`);
    console.log(`[页面] 提示框 warn ${stats.warn} / tip ${stats.tip} / done ${stats.done}`
      + ` ｜ 图片 ${stats.images} 张，加载失败 ${stats.broken} 张`);
    if (stats.broken > 0) {
      console.log('       ^ 有图没加载出来，先查相对路径（HTML 必须在仓库根）');
    }

    // ---- 截图（可选）----
    if (shotDir) {
      fs.mkdirSync(shotDir, { recursive: true });
      // 模拟 print 媒体：book.css 的折行规则只在打印时生效，
      // 不模拟的话截出来的是屏幕版，看不到真实效果
      await cdp.send('Emulation.setEmulatedMedia', { media: 'print' }, sessionId);
      // A4 宽 210mm - 左右各 16mm 页边距 = 178mm ≈ 673px(96dpi)
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width: 673, height: 950, deviceScaleFactor: 1.5, mobile: false,
      }, sessionId);

      const targets = scrollArgs.length ? scrollArgs : ['y:0'];
      for (let i = 0; i < targets.length; i++) {
        const target = targets[i];
        let expr;
        if (target.startsWith('sel:')) {
          const sel = JSON.stringify(target.slice(4));
          // 找不到就返回 -1，跳过这一张，而不是傻乎乎地截当前视口
          expr = `(() => {
            const el = document.querySelector(${sel});
            if (!el) return -1;
            window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 24);
            return 1;
          })()`;
        } else {
          expr = `(window.scrollTo(0, ${Number(target.replace(/^y:/, ''))}), 1)`;
        }
        const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true },
          sessionId);
        if (r.result.value === -1) {
          console.log(`[截图] 跳过 ${target} —— 页面上没有匹配的元素`);
          continue;
        }
        await new Promise((r2) => setTimeout(r2, 250));
        const shot = await cdp.send('Page.captureScreenshot', { format: 'png' }, sessionId);
        const tag = target.replace(/[^A-Za-z0-9_.-]/g, '_');
        const out = path.join(shotDir, `shot-${String(i).padStart(2, '0')}-${tag}.png`);
        fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
        console.log(`[截图] ${out}`);
      }
      await cdp.send('Emulation.clearDeviceMetricsOverride', {}, sessionId);
    }

    // ---- 出 PDF ----
    // preferCSSPageSize: 纸张尺寸以 book.css 的 `@page { size: A4 }` 为准，
    // 免得这里和 CSS 两处各写一份 A4，改了一处忘另一处。
    const { data } = await cdp.send('Page.printToPDF', {
      printBackground: true,
      preferCSSPageSize: true,
      displayHeaderFooter: FOOTER !== '',
      headerTemplate: '<div></div>',
      footerTemplate: FOOTER,
      margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
    }, sessionId);

    fs.mkdirSync(path.dirname(pdfPath), { recursive: true });
    fs.writeFileSync(pdfPath, Buffer.from(data, 'base64'));
    const kb = (fs.statSync(pdfPath).size / 1024).toFixed(0);
    console.log(`[PDF ] ${pdfPath}（${kb} KB）`);
    return 0;
  } finally {
    if (ws) ws.close();
    proc.kill();
    // Edge 有时退不干净，留一个临时 profile 无所谓，但能删就删
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (_) { /* 忽略 */ }
  }
}

main().then(
  (code) => process.exit(code),
  (err) => { console.error(`[失败] ${err.message}`); process.exit(1); },
);
