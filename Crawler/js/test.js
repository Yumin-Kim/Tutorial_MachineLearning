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
    await page.goto("https://www.google.com/search?q=%EC%84%9D%ED%83%91&tbm=isch&ved=2ahUKEwjep-ntruHuAhV3zosBHQW2BnAQ2-cCegQIABAA&oq=%EC%84%9D%ED%83%91&gs_lcp=CgNpbWcQAzIECCMQJzICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyBggAEAUQHjIGCAAQBRAeUKIWWJEZYLoaaABwAHgAgAF8iAHNA5IBAzAuNJgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=YOQkYN73Afecr7wPheyagAc&bih=969&biw=1920")
    // await autoScroll(page);

    // const IMAGE_SELECTOR = ".rg_i"
    // let imageHref = await page.evaluate((sel)=>{
    //     return document.querySelectorAll(sel);
    // },IMAGE_SELECTOR);
    const imageURL = await page.$eval(".rg_i", img => img.src);
    const imageURLs = await page.$$eval(".rg_i", imgsAll => imgsAll.map(img => img.src));
    // download(imageURL,filepath,(filename)=>{
    //     console.log("Download complate for "+ filename);
    // })
    // fs.readdir("Tower", (error, file) => {
    //     if (error) {
    //         fs.mkdir("Tower", () => {
    //             console.log("mkdir Tower")
    //             return;
    //         })
    //     }
    //     else {

    //     }
    // })
        download(imageURLs ,await page);
    

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