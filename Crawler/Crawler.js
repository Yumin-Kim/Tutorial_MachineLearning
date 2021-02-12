var Scraper = require('images-scraper');

var request = require('request');
var fs = require('fs');

const google = new Scraper({
    puppeteer:{},
    tbs:{
        isz : "etc",
    }
});
let num = 0;
(async () => {
    const results = await google.scrape('한국 석탑', 100);
    await results.forEach((val)=>{
        download( val.url,`new8_${num}.png`, function () {
            console.log('done');
        }) 
        num++;
    })

})();
var download = function (uri, filename, callback) {
    if(uri.match(/x-raw-image/g)===null){
        request.head(uri, function (err, res, body) {
            request(uri).pipe(fs.createWriteStream(filename)).on('close', callback);
        });
    }
};