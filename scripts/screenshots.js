// Regenerates the README screenshots from a running, seeded MemBrain.
//
//   python -m app.cli seed-demo --workspace demo     # in backend/, against a scratch database
//   WORKSPACE=demo uvicorn app.main:app --port 8099
//   npm install --no-save puppeteer-core && node scripts/screenshots.js
//
// Env: BASE_URL (default http://localhost:8099/), BROWSER (path to Chrome or Edge), SHOOT_ASK=1 to include
// a cited answer (needs ANTHROPIC_API_KEY on the server).
const path = require('path');
const puppeteer = require('puppeteer-core');
const OUT = path.join(__dirname, '..', 'docs', 'screenshots');
const BASE = process.env.BASE_URL || 'http://localhost:8099/';
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const browser = await puppeteer.launch({
    executablePath: process.env.BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true, args: ['--hide-scrollbars'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  const open = async panel => { await page.goto(BASE + '#' + panel, { waitUntil: 'networkidle0' }); await sleep(900); };

  await open('search');
  await page.type('#search-q', 'what went wrong with payments');
  await page.evaluate(() => doSearch());
  await sleep(4000);  // let the result toast clear
  await page.screenshot({ path: `${OUT}/search.png` });

  if (process.env.SHOOT_ASK) {
    await open('qa');
    await page.type('#qa-q', 'What caused the checkout outage, and what are the follow-up actions?');
    await page.evaluate(() => sendQA());
    await page.waitForFunction(() => document.querySelectorAll('#qa-chat .cite').length > 0, { timeout: 90000 });
    await sleep(500);
    await page.screenshot({ path: `${OUT}/ask.png` });
  }

  await open('graph');
  await sleep(7000);
  const box = await (await page.$('#graph-canvas')).boundingBox();
  // Select the node nearest the centre for the details card.
  let opened = false;
  for (let r = 0; r < 260 && !opened; r += 6) {
    for (let a = 0; a < Math.PI * 2 && !opened; a += Math.PI / 12) {
      await page.mouse.click(box.x + box.width / 2 + Math.cos(a) * r, box.y + box.height / 2 + Math.sin(a) * r);
      opened = await page.$eval('#graph-info', el => !el.hidden);
    }
  }
  await page.mouse.move(box.x + 5, box.y + 5);
  await sleep(400);
  await page.screenshot({ path: `${OUT}/graph.png` });

  await open('decay');
  await page.screenshot({ path: `${OUT}/decay.png` });

  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1, isMobile: true, hasTouch: true });
  await open('browse');
  await page.screenshot({ path: `${OUT}/mobile.png` });

  console.log('screenshots written');
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
