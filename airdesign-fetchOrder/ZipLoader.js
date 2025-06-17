import unzip from 'unzip-js';

class zipLoader {
    constructor() {
        this.unzipData = [];
        this.entryCount = 0;
        this.readCount = 0;
    }

    loadAsync = async (blob, progress = () => { }) => {
        return await new Promise((resolve, reject) => {
            try {
                this.load(blob, progress, resolve);
            } catch (error) {
                reject(error);
            }
        });
    }

    load = (blob, progress = () => { }, handleUnzipObjects = () => { }) => {
        const data = [];

        try {
            unzip(blob, (err, zipFile) => {
                if (err) throw err;

                zipFile.readEntries((err, entries) => {
                    if (err) throw err;

                    entries.forEach(entry => {
                        this.entryCount++;
                        zipFile.readEntryData(entry, false, (err, readStream) => {
                            if (err) throw err;

                            readStream.on('data', chunk => {
                                data.push({
                                    "name": entry.name,
                                    "data": chunk
                                });
                            });

                            readStream.on('end', () => {
                                this.readCount++;
                                progress(Math.floor(this.readCount / entries.length * 100));
                                if (this.entryCount != this.readCount) return;

                                this.concatOriginalfile(data, handleUnzipObjects);
                            });
                        });
                    });
                });
            });
        } catch (error) {
            throw error;
        }
    }

    concatOriginalfile = (unzipData, handleUnzipObjects) => {
        const files = [];
        for (let i = 0; i < unzipData.length; i++) {
            if (files.includes(unzipData[i].name)) continue;
            files.push(unzipData[i].name);
        }

        for (let i = 0; i < files.length; i++) {
            const arrayBlob = [];
            const data = unzipData.filter(item => item.name == files[i]);

            for (let j = 0; j < data.length; j++) {
                arrayBlob.push(data[j].data);
            }

            const result = this.concatArray(arrayBlob);
            this.unzipData[files[i]] = new Blob([result]);
        }
        handleUnzipObjects(this.unzipData);
    }

    concatArray = (arrays) => {
        // sum of individual array lengths
        let totalLength = arrays.reduce((acc, value) => acc + value.length, 0);
        if (!arrays.length) return null;
        let result = new Uint8Array(totalLength);
        // for each array - copy it over result
        // next array is copied right after the previous one
        let length = 0;
        for (let array of arrays) {
            result.set(array, length);
            length += array.length;
        }
        return result;
    }
}

export default zipLoader;