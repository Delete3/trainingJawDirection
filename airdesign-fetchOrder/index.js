import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Lodash from 'lodash';
import axios from 'axios';
import ZipLoader from './ZipLoader.js';
import AdmZip from 'adm-zip';
import { pidArray } from './PidArray.js'

const url = 'https://test-airdental.inteware.com.tw';
const account = 'labtest@inteware.com.tw';
const password = 'intewaretest5678';

// const url = 'https://test-illusion.inteware.com.tw';
// const account = 'superusertest@inteware.com.tw';
// const password = 'inteware1234test1234';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.resolve(__dirname, 'data');

const createFolder = (folderPath) => {
    try {
        fs.accessSync(folderPath);
        console.log('資料夾已經存在');
    } catch (err) {
        fs.mkdirSync(folderPath);
        console.log('資料夾創建成功');
    }
}

const extractFilename = (disposition) => {
    let match = disposition.match(/filename="([^"]+)"/);
    if (match) return match[1];

    match = disposition.match(/filename\*=([a-zA-Z0-9\-]+)''(.+)/);
    if (match) return decodeURIComponent(match[2]);

    return null;
}

const downloadFile = (response, filename, filepath) => {
    try {
        const disposition = response.headers['content-disposition'];
        const fileNameWithExtension = extractFilename(disposition);
        const extension = fileNameWithExtension.split('.').pop();
        const fullFilepath = path.resolve(filepath, `${filename}.${extension}`);

        fs.writeFileSync(fullFilepath, response.data);
        console.log(`下載完成：${fullFilepath}`)
    } catch (error) {
        console.error('下載過程中發生錯誤:', error);
    }
}

(async () => {
    const loginResponse = await axios({
        method: 'post',
        url: `${url}/api/login`,
        data: { account, password }
    });
    const cookies = loginResponse.headers['set-cookie'];
    const axiosWithCookies = axios.create({
        withCredentials: true,
        headers: {
            'Cookie': cookies
        }
    });

    const loadProjectFile = async pid => {
        try {
            const folderPath = path.join(dataPath, pid);
            createFolder(folderPath);

            const projectDataResponse = await axiosWithCookies.get(`${url}/api/v2/airdesign/project/${pid}`);

            const upperFid = Lodash.get(projectDataResponse.data, 'maxillaSTLID.ckey', null);
            if (!upperFid) return;
            const upperResponse = await axiosWithCookies.get(`${url}/api/v2/airdesign/file/model/${upperFid}`, { responseType: 'arraybuffer' });
            downloadFile(upperResponse, 'upper', folderPath);

            const lowerFid = Lodash.get(projectDataResponse.data, 'mandibleSTLID.ckey', null);
            if (!lowerFid) return;
            const lowerResponse = await axiosWithCookies.get(`${url}/api/v2/airdesign/file/model/${lowerFid}`, { responseType: 'arraybuffer' });
            downloadFile(lowerResponse, 'lower', folderPath);

            const crownDataFid = Lodash.get(projectDataResponse.data, 'crownData', null);
            if (!crownDataFid) return;

            const crownDataResponse = await axiosWithCookies.get(`${url}/api/v2/airdesign/file/json/${crownDataFid}`);
            const originalAIFileFid = Lodash.get(crownDataResponse.data, 'originalAIFileFid', null);
            if (!originalAIFileFid) return;
            const originalAIFileResponse = await axiosWithCookies.get(`${url}/api/v2/airdesign/file/zip/${originalAIFileFid}`, { responseType: 'arraybuffer' });
            downloadFile(originalAIFileResponse, 'originalAIFile', folderPath);

            // 解壓縮
            const originalAIFileBuffer = Buffer.from(originalAIFileResponse.data);
            const zip = new AdmZip(originalAIFileBuffer);

            for (const zipEntry of zip.getEntries()) {
                const { entryName } = zipEntry;
                if (entryName.includes('matrix') && entryName.includes('.json')) {
                    const dataBuffer = zipEntry.getData();
                    const fullFilepath = path.resolve(folderPath, entryName);
                    fs.writeFileSync(fullFilepath, dataBuffer);
                }
            }
        } catch (error) {
            console.log(error);
        }
    }

    try {
        // await loadProjectFile('67eb3a6fc5387fddf06be80c');
        // console.log(pidArray)
        for(const pid of pidArray){
            await loadProjectFile(pid)
        }
    } catch (err) {
        console.log(err);
    } finally {
        await axiosWithCookies({
            method: 'delete',
            url: `${url}/api/logout`
        });
    }
})();