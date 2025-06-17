import fs from 'fs';
import path from 'path';
import AdmZip from 'adm-zip';

const rootFolderPath = './data/200';
const folderNameArray = fs.readdirSync(rootFolderPath);

for (const folderName of folderNameArray) {
    const zipFilePath = path.join(rootFolderPath, folderName, 'originalAIFile.zip');
    const bufferData = fs.readFileSync(zipFilePath);

    const zip = new AdmZip(bufferData);
    for (const zipEntry of zip.getEntries()) {
        const { entryName } = zipEntry;
        if (entryName.includes('margin') && entryName.includes('.pts')) {
            const dataBuffer = zipEntry.getData();
            const fullFilepath = path.resolve(rootFolderPath, folderName, entryName);
            fs.writeFileSync(fullFilepath, dataBuffer);
        }
    }
}