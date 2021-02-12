const https = require("https");
const http = require("http");
const fs = require("fs");
const path = require("path");
const URL = require("url").URL;

function download (url , filepath ,callback){
    const userURL = new URL(url);
    const requestCaller = userURL.protocol === "http:"? http : https;
    const filename = path.basename(url);

    const req = requestCaller.get(url,(res)=>{
        const fileStream = fs.createWriteStream(path.resolve(filepath , filename));
        req.pipe(filename);

        fileStream.on("error",(error)=>{
            console.log("Error writting to the stream");
            console.log(error);
        })

        fileStream.on("close",()=>{
            callback(filename);
        })

        fileStream.on("finish",()=>{
            fileStream.close();
        })
    });

    req.on("error",(error)=>{
        console.log("Error downloading the file");
        console.log(error); 
    })
}

module.exports.download = download;