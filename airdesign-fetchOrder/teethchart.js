/**************************************************
 File: teethchart.js
 Name: Teeth chart process
 Explain: 齒位表處理
****************************************By QQBoxy*/
/*jshint node: true, expr: true, esversion: 6, browser: true*/

const TeethChart = function () {
    /**齒位標準
     * [UNS] Unsversal Numbering System 通用記錄法
     * [FDI] Fédération Dentaire Internationale 國際牙科聯盟
     * [Pal] Palmer notation 帕爾默牙位表示法
     */
    this.rules = [
        // 恆齒
        // 上顎 - 右
        { id: 0, uns: "1", fdi: "18", pal: "8┘" },
        { id: 1, uns: "2", fdi: "17", pal: "7┘" },
        { id: 2, uns: "3", fdi: "16", pal: "6┘" },
        { id: 3, uns: "4", fdi: "15", pal: "5┘" },
        { id: 4, uns: "5", fdi: "14", pal: "4┘" },
        { id: 5, uns: "6", fdi: "13", pal: "3┘" },
        { id: 6, uns: "7", fdi: "12", pal: "2┘" },
        { id: 7, uns: "8", fdi: "11", pal: "1┘" },
        // 上顎 - 左
        { id: 8, uns: "9", fdi: "21", pal: "└1" },
        { id: 9, uns: "10", fdi: "22", pal: "└2" },
        { id: 10, uns: "11", fdi: "23", pal: "└3" },
        { id: 11, uns: "12", fdi: "24", pal: "└4" },
        { id: 12, uns: "13", fdi: "25", pal: "└5" },
        { id: 13, uns: "14", fdi: "26", pal: "└6" },
        { id: 14, uns: "15", fdi: "27", pal: "└7" },
        { id: 15, uns: "16", fdi: "28", pal: "└8" },
        // 下顎 - 左
        { id: 16, uns: "17", fdi: "38", pal: "┌8" },
        { id: 17, uns: "18", fdi: "37", pal: "┌7" },
        { id: 18, uns: "19", fdi: "36", pal: "┌6" },
        { id: 19, uns: "20", fdi: "35", pal: "┌5" },
        { id: 20, uns: "21", fdi: "34", pal: "┌4" },
        { id: 21, uns: "22", fdi: "33", pal: "┌3" },
        { id: 22, uns: "23", fdi: "32", pal: "┌2" },
        { id: 23, uns: "24", fdi: "31", pal: "┌1" },
        // 下顎 - 右
        { id: 24, uns: "25", fdi: "41", pal: "1┐" },
        { id: 25, uns: "26", fdi: "42", pal: "2┐" },
        { id: 26, uns: "27", fdi: "43", pal: "3┐" },
        { id: 27, uns: "28", fdi: "44", pal: "4┐" },
        { id: 28, uns: "29", fdi: "45", pal: "5┐" },
        { id: 29, uns: "30", fdi: "46", pal: "6┐" },
        { id: 30, uns: "31", fdi: "47", pal: "7┐" },
        { id: 31, uns: "32", fdi: "48", pal: "8┐" },
        // 乳齒
        // 上顎 - 右
        { id: 32, uns: "A", fdi: "55", pal: "E┘" },
        { id: 33, uns: "B", fdi: "54", pal: "D┘" },
        { id: 34, uns: "C", fdi: "53", pal: "C┘" },
        { id: 35, uns: "D", fdi: "52", pal: "B┘" },
        { id: 36, uns: "E", fdi: "51", pal: "A┘" },
        // 上顎 - 左
        { id: 37, uns: "F", fdi: "61", pal: "└A" },
        { id: 38, uns: "G", fdi: "62", pal: "└B" },
        { id: 39, uns: "H", fdi: "63", pal: "└C" },
        { id: 40, uns: "I", fdi: "64", pal: "└D" },
        { id: 41, uns: "J", fdi: "65", pal: "└E" },
        // 下顎 - 左
        { id: 42, uns: "K", fdi: "75", pal: "┌E" },
        { id: 43, uns: "L", fdi: "74", pal: "┌D" },
        { id: 44, uns: "M", fdi: "73", pal: "┌C" },
        { id: 45, uns: "N", fdi: "72", pal: "┌B" },
        { id: 46, uns: "O", fdi: "71", pal: "┌A" },
        // 下顎 - 右
        { id: 47, uns: "P", fdi: "81", pal: "A┐" },
        { id: 48, uns: "Q", fdi: "82", pal: "B┐" },
        { id: 49, uns: "R", fdi: "83", pal: "C┐" },
        { id: 50, uns: "S", fdi: "84", pal: "D┐" },
        { id: 51, uns: "T", fdi: "85", pal: "E┐" },
    ];
};

//FDI Sort
TeethChart.prototype.fdiSort = function (fdis, callback) {
    fdis.sort((a, b) => {
        const ra = this.rules.find((r) => r.fdi == a);
        const rb = this.rules.find((r) => r.fdi == b);
        return ra.id - rb.id;
    });
    if (callback) callback(fdis);
    return fdis;
};

//FDI Sorter
TeethChart.prototype.Sorter = function (a, b) {
    const ra = this.rules.find((r) => r.fdi == a);
    const rb = this.rules.find((r) => r.fdi == b);
    return ra.id - rb.id;
};

//UNS 轉 FDI
TeethChart.prototype.unsToFdi = function (uns, callback) {
    const result = this.rules.find(rule => rule.uns == uns);
    if (result) {
        if (callback) callback(result.fdi);
        return result.fdi;
    } else {
        if (callback) callback(null);
        return null;
    }
};

//FDI 轉 UNS
TeethChart.prototype.fdiToUns = function (fdi, callback) {
    const result = this.rules.find(rule => rule.fdi == fdi);
    if (result) {
        if (callback) callback(result.uns);
        return result.uns;
    } else {
        if (callback) callback(null);
        return null;
    }
};

//UNS 轉 ID
TeethChart.prototype.unsToId = function (uns, callback) {
    const result = this.rules.find(rule => rule.uns == uns);
    if (result) {
        if (callback) callback(result.id);
        return result.id;
    } else {
        if (callback) callback(null);
        return null;
    }
};

//FDI 轉 ID
TeethChart.prototype.fdiToId = function (fdi, callback) {
    const result = this.rules.find(rule => rule.fdi == fdi);
    if (result) {
        if (callback) callback(result.id);
        return result.id;
    } else {
        if (callback) callback(null);
        return null;
    }
};

//ID 轉 UNS
TeethChart.prototype.idToUns = function (id, callback) {
    const result = this.rules.find(rule => rule.id == id);
    if (result) {
        if (callback) callback(result.uns);
        return result.uns;
    } else {
        if (callback) callback(null);
        return null;
    }
};

//ID 轉 FDI
TeethChart.prototype.idToFdi = function (id, callback) {
    const result = this.rules.find(rule => rule.id == id);
    if (result) {
        if (callback) callback(result.fdi);
        return result.fdi;
    } else {
        if (callback) callback(null);
        return null;
    }
};

// Initial
const teethChartObj = new TeethChart();
export default teethChartObj;