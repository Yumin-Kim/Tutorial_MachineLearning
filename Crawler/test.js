const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const { download } = require("./DownLoader");

const filepath = path.resolve(__dirname , "Tower")

async function main() {

    const browser = await puppeteer.launch({
        headless: false
    })
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 1200 })
    await page.goto("https://www.pinterest.co.kr/search/pins/?q=%EB%AA%A9%ED%83%91&rs=typed&term_meta[]=%EB%AA%A9%ED%83%91%7Ctyped")
    // await autoScroll(page);

    // const IMAGE_SELECTOR = ".rg_i"
    // let imageHref = await page.evaluate((sel)=>{
    //     return document.querySelectorAll(sel);
    // },IMAGE_SELECTOR);
    // const imageURL = await page.$eval(".rg_i", img => img.src);
    const imageURLs = await page.$$eval(".MIw", imgsAll => imgsAll.map(img => img.src));
  
    imageURLs.forEach(imageURL =>{
        download(imageURL,filepath,(filename)=>{
          console.log("Downloading complate for"+filename);  
        })
    })

    //     browser.close();
}

async function autoScroll(page) {
    await page.evaluate(async () => {
        await new Promise((resolve, reject) => {
            var totalHeight = 0;
            var distance = 100;
            var timer = setInterval(() => {
                var scrollHeight = document.body.scrollHeight;
                window.scrollBy(0, distance);
                totalHeight += distance;

                if (totalHeight >= scrollHeight) {
                    clearInterval(timer);
                    resolve();
                }
            }, 100);
        });
    });
}

main();