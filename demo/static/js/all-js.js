// Special thanks to:
//   - Zhanwei Xu
//   - Zhangliang Sun

const sliderValueLimit = 10
let glandLengths = [];
let glandWidths = [];
let glandAreas = [];
let glandCurvatures = [];
let totalGlandNum = 0;
let glandLabels = null;
let pseudoScribbleEnabled = true;
let foregroundPointsLast = [];
let backgroundPointsLast = [];
let segHistory = []; // 用来保存每次分割完成时的点击列表
let revert = false


const fileInput = document.getElementById('imageUpload');
let img = new Image();

fileInput.addEventListener('change', (e) => {
const file = e.target.files[0];

if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
        img.src = event.target.result;
    }
    reader.readAsDataURL(file);
}
});


const sliderInput = document.querySelectorAll("input")[1];
const sliderLabel = document.querySelectorAll("label")[1];

const initValue = 5;
sliderInput.value = (initValue / sliderValueLimit) * 100;
sliderLabel.innerHTML = initValue;
let currentImage = img;

sliderInput.addEventListener("input", event => {
    const sliderValue = Number(sliderInput.value) / 100;
    sliderInput.style.setProperty("--thumb-rotate", `${sliderValue * 720}deg`);
    sliderLabel.innerHTML = Math.round(sliderValue * sliderValueLimit);
    updatePointsRadius(sliderValue);
});

function updatePointsRadius(sliderValue) {
    // 计算新的半径
    const newRadius = sliderValue * sliderValueLimit;  // 这里假设半径随 sliderValue 变化

    // 清除当前的绘制
    sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);

    // 重新绘制图像（如果你需要重新绘制原始图像）
    sourceCtx.drawImage(currentImage, 0, 0, img.width, img.height);

    // 绘制前景点
    sourceCtx.fillStyle = 'green';
    for (let i = 0; i < foregroundPoints.length; i++) {
        let x = foregroundPoints[i].x;
        let y = foregroundPoints[i].y;

        sourceCtx.beginPath();
        sourceCtx.arc(x, y, newRadius, 0, 2 * Math.PI);
        sourceCtx.fill();
    }

    // 绘制背景点
    sourceCtx.fillStyle = 'red';
    for (let i = 0; i < backgroundPoints.length; i++) {
        let x = backgroundPoints[i].x;
        let y = backgroundPoints[i].y;

        sourceCtx.beginPath();
        sourceCtx.arc(x, y, newRadius, 0, 2 * Math.PI);
        sourceCtx.fill();
    }
}


const pointType = document.getElementById('pointType');
const sendBtn = document.getElementById('sendBtn');
const resetBtn = document.getElementById('resetBtn');
const revertBtn = document.getElementById('revertBtn');
const clearBtn = document.getElementById('clearBtn');
const saveBtn = document.getElementById('saveBtn');
const exportBtn = document.getElementById('exportExcelBtn');
const sourceCanvas = document.getElementById('sourceCanvas');
const resultCanvas = document.getElementById('resultCanvas');
const sourceCtx = sourceCanvas.getContext('2d');
const resultCtx = resultCanvas.getContext('2d');
const pxToMmInput = document.getElementById('px-mm-ratio');
const glandNumber = document.getElementById('gland-number');
const glandLength = document.getElementById('gland-length');
const glandWidth = document.getElementById('gland-width');
const glandArea = document.getElementById('gland-area');
const glandCurvature = document.getElementById('gland-curvature');
const patientName = document.getElementById('name');
const patientSex = document.getElementById('sex');
const patientAge = document.getElementById('age');
const patientEye = document.getElementById('eye');
const examDate = document.getElementById('exam-date');
const Meiboscore = document.getElementById('meiboscore');
const overallGlandNumber = document.getElementById('overall-gland-number');
const averageGlandLength = document.getElementById('average-gland-length');
const averageGlandWidth = document.getElementById('average-gland-width');
const AverageGlandCurvature = document.getElementById('average-gland-curvature');
const averageGlandArea = document.getElementById('average-gland-area');
const toggle = document.getElementById('pseudoScribbleToggle');
const scribbleCountInput = document.getElementById("scribbleCount");


var foregroundPoints = [];
var backgroundPoints = [];

img.onload = () => {
    sourceCanvas.width = img.width;
    sourceCanvas.height = img.height;
    resultCanvas.width = img.width;
    resultCanvas.height = img.height;
    sourceCtx.drawImage(img, 0, 0, img.width, img.height);
    currentImage = img;
};

sourceCanvas.addEventListener('click', (e) => {
    var x = e.offsetX / sourceCanvas.clientWidth * sourceCanvas.width;
    var y = e.offsetY / sourceCanvas.clientHeight * sourceCanvas.height;
    const sliderValue = Number(sliderInput.value) / 100 * sliderValueLimit;
    if (pointType.value === 'foreground') {
        foregroundPoints.push({ x, y });
        sourceCtx.fillStyle = 'green';
    } else {
        backgroundPoints.push({ x, y });
        sourceCtx.fillStyle = 'red';
    }
    sourceCtx.closePath();
    sourceCtx.beginPath();
    sourceCtx.arc(x, y, sliderValue, 0, 2 * Math.PI);
    sourceCtx.fill();
});

sourceCanvas.addEventListener('mousemove', (e) => {
     console.log("move");
    if (!glandLabels) return;

    // 映射到 Canvas 原尺寸坐标
    const xCanvas = e.offsetX / sourceCanvas.clientWidth  * sourceCanvas.width;
    const yCanvas = e.offsetY / sourceCanvas.clientHeight * sourceCanvas.height;

    // glandLabels 的 H × W
    const H = glandLabels.length;
    const W = glandLabels[0].length;

    // 转为 mask 里的索引
    const ix = Math.floor(xCanvas * W / sourceCanvas.width);
    const iy = Math.floor(yCanvas * H / sourceCanvas.height);

    if (iy < 0 || iy >= H || ix < 0 || ix >= W) return;

    const gid = glandLabels[iy][ix];   // 腺体编号

    if (gid > 0) {
        glandNumber.textContent = gid;

        // label 是从 1 开始，数组索引从 0
        const idx = gid - 1;

        const pxToMm = parseFloat(pxToMmInput.value);

        const length = glandLengths[idx] ?? null;
        const width  = glandWidths[idx] ?? null;
        const area   = glandAreas[idx] ?? null;
        const curv   = glandCurvatures[idx] ?? null;

        glandLength.textContent =
            length != null ? (length * pxToMm).toFixed(3) : '--';

        glandWidth.textContent =
            width != null ? (width * pxToMm).toFixed(3) : '--';

        glandArea.textContent =
            area != null ? (area * pxToMm * pxToMm).toFixed(3) : '--';

        glandCurvature.textContent =
            curv != null ? curv.toFixed(3) : '--';
    } else {
        // 背景时清空
        glandNumber.textContent = '--';
        glandLength.textContent = '--';
        glandWidth.textContent = '--';
        glandArea.textContent = '--';
        glandCurvature.textContent = '--';
    }
});


sendBtn.addEventListener('click', async () => {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    const sliderValue = Number(sliderInput.value) / 100 * sliderValueLimit;
    segHistory.push({
        fg: foregroundPoints.map(p => ({ ...p })),
        bg: backgroundPoints.map(p => ({ ...p })),
    });
    revert = false
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    tempCtx.drawImage(img, 0, 0, img.width, img.height);
    let scribbleCount  = Number(scribbleCountInput.value)
    var url = '/submit_post_message';
    const data = {
        image: tempCanvas.toDataURL('image/jpeg'),
        foregroundPoints,
        backgroundPoints,
        sliderValue,
        pseudoScribbleEnabled,
        scribbleCount
    };

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
        
    const result = await response.json();
    // 形态学参数分析
    glandLengths = result.gland_lengths;
    glandWidths = result.gland_widths;
    glandAreas = result.gland_areas;
    glandCurvatures = result.gland_curvatures;
    totalGlandNum = result.total_gland_num;
    glandLabels = result.labels;
    updateGlobalMetrics();

    const resultImg = new Image();
    resultImg.src = "data:image/jpeg;base64," + result.output_image;
    resultImg.onload = () => {
        currentImage = resultImg;
        sourceCtx.drawImage(resultImg, 0, 0, resultImg.width, resultImg.height);

        sourceCtx.fillStyle = 'green';
        for (let i = 0; i < foregroundPoints.length; i++) {
            let x = foregroundPoints[i].x;
            let y = foregroundPoints[i].y;
            
            sourceCtx.beginPath();
            sourceCtx.arc(x, y, sliderValue, 0, 2 * Math.PI);
            sourceCtx.fill();
        }
    
        sourceCtx.fillStyle = 'red';
        for (let i = 0; i < backgroundPoints.length; i++) {
            let x = backgroundPoints[i].x;
            let y = backgroundPoints[i].y;
            
            sourceCtx.beginPath();
            sourceCtx.arc(x, y, sliderValue, 0, 2 * Math.PI);
            sourceCtx.fill();
        }
    }
    const resultMask = new Image();
    resultMask.src = "data:image/jpeg;base64," + result.output_mask;
    resultMask.onload = () => {
        resultCtx.drawImage(resultMask, 0, 0, resultMask.width, resultMask.height);
    }
});

resetBtn.addEventListener('click', () => {
    const sliderValue = Number(sliderInput.value) / 100 * sliderValueLimit;

    if (pointType.value === 'foreground') {
        foregroundPoints.pop();
    } else {
        backgroundPoints.pop();
    }

    sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    sourceCtx.drawImage(currentImage, 0, 0, img.width, img.height);

    sourceCtx.fillStyle = 'green';
    for (let i = 0; i < foregroundPoints.length; i++) {
        const p = foregroundPoints[i];
        sourceCtx.beginPath();
        sourceCtx.arc(p.x, p.y, sliderValue, 0, 2 * Math.PI);
        sourceCtx.fill();
    }

    sourceCtx.fillStyle = 'red';
    for (let i = 0; i < backgroundPoints.length; i++) {
        const p = backgroundPoints[i];
        sourceCtx.beginPath();
        sourceCtx.arc(p.x, p.y, sliderValue, 0, 2 * Math.PI);
        sourceCtx.fill();
    }
});

revertBtn.addEventListener("click", async () => {

    const response = await fetch("/revert", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    });

    const data = await response.json();

    if (revert || !data || (!data.output_mask && !data.output_image)) {
        alert("Can only revert to the previous step.");
        return;
    }

    // === 显示 Seg mask（图内叠 mask + scribble 已经在后端完成）===
    const resultImg = new Image();
    resultImg.src = "data:image/jpeg;base64," + data.output_image;

    resultImg.onload = () => {
        currentImage = resultImg
        revert = true
        segHistory.pop();
        const lastSeg = segHistory[segHistory.length - 1];
        foregroundPoints = lastSeg.fg.map(p => ({ ...p }));
        backgroundPoints = lastSeg.bg.map(p => ({ ...p }));

        // 清理原图
        sourceCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

        // 绘制生成图
        sourceCtx.drawImage(resultImg, 0, 0, resultImg.width, resultImg.height);

        // ====== 重新绘制用户点击点 ======
        const sliderValue = Number(sliderInput.value) / 100 * sliderValueLimit;

        // foreground → green
        sourceCtx.fillStyle = "green";
        for (let i = 0; i < foregroundPoints.length; i++) {
            let x = foregroundPoints[i].x;
            let y = foregroundPoints[i].y;

            sourceCtx.beginPath();
            sourceCtx.arc(x, y, sliderValue, 0, 2 * Math.PI);
            sourceCtx.fill();
        }

        // background → red
        sourceCtx.fillStyle = "red";
        for (let i = 0; i < backgroundPoints.length; i++) {
            let x = backgroundPoints[i].x;
            let y = backgroundPoints[i].y;

            sourceCtx.beginPath();
            sourceCtx.arc(x, y, sliderValue, 0, 2 * Math.PI);
            sourceCtx.fill();
        }
    };

    const resultMask = new Image();
    resultMask.src = "data:image/jpeg;base64," + data.output_mask;
    resultMask.onload = () => {
        resultCtx.drawImage(resultMask, 0, 0, resultMask.width, resultMask.height);
    }
});



clearBtn.addEventListener('click', async () => {
    sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    foregroundPoints = []
    backgroundPoints = []
    glandLengths = [];
    glandWidths = [];
    glandAreas = [];
    glandCurvatures = [];
    totalGlandNum = 0;
    glandLabels = null;
    updateGlobalMetrics()
    patientName.value = "";
    patientSex.value = "";
    patientAge.value = "";
    patientEye.value = "";
    examDate.value = "";
    Meiboscore.value = "";
    var url = '/clear_post_message';
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });
});

saveBtn.addEventListener("click", saveCanvasAsPNG);


exportBtn.addEventListener('click', async () => {
    const pxToMm = parseFloat(pxToMmInput.value) || 0;
    // 1) patient info
    const data = {
        patient: {
            name: patientName.value,
            sex: patientSex.value,
            age: patientAge.value,
            eye: patientEye.value,
            exam_date: examDate.value,
            meiboscore: Meiboscore.value,
        },

        // 2) single gland morphology arrays
        glands: {
            lengths: glandLengths,
            widths: glandWidths,
            areas: glandAreas,
            curvatures: glandCurvatures,
        },

        // 3) overall metrics
        overall: {
            number: overallGlandNumber.textContent,
            avg_length: averageGlandLength.textContent,
            avg_width: averageGlandWidth.textContent,
            avg_curvature: AverageGlandCurvature.textContent,
            avg_area: averageGlandArea.textContent,
        },
        px_to_mm: pxToMm
    };

    const response = await fetch("/export_excel", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        alert("Failed to export Excel");
        return;
    }

    // === 处理返回的 Excel Blob ===
    const blob = await response.blob();
    await saveExcelWithPicker(blob);

});


var colorPicker = document.getElementById("colorPicker");
var colorPickerBtn = document.getElementById("colorPickerBtn");

function showColorPicker() {
    if (colorPicker.style.display === "none") {
        colorPicker.style.display = "block";
    } else {
        colorPicker.style.display = "none";
    }
}

// https://www.npmjs.com/package/vue-color
// https://github.com/xiaokaike/vue-color
// https://github.com/xiaokaike/vue-color#readme
Vue.component("color-picker", VueColor.Sketch), new Vue({
  el: "#vue_sketch_picker",
  data: function() {
    return {
      colors: {
        rgba: {
            r: 128,
            g: 0,
            b: 0,
            a: 0.6,
        }
      }
    }
  },
  methods: {
    updateValue: async function(selectedColor) {
        console.log(selectedColor, selectedColor.rgba.r, selectedColor.rgba.g, selectedColor.rgba.b, selectedColor.rgba.a)
        
        const data = {
            color_r: selectedColor.rgba.r,
            color_g: selectedColor.rgba.g,
            color_b: selectedColor.rgba.b,
            color_a: selectedColor.rgba.a,
        };
        
        const response = await fetch('/change_color_post_message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        

        const result = await response.json();
        const resultImg = new Image();

        resultImg.src = "data:image/jpeg;base64," + result.output_image;
        resultImg.onload = () => {
            sourceCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
            sourceCtx.drawImage(resultImg, 0, 0, resultImg.width, resultImg.height);
            const sliderValue = Number(sliderInput.value) / 100 * sliderValueLimit;
            sourceCtx.fillStyle = 'green';
            for (let i = 0; i < foregroundPoints.length; i++) {
                let x = foregroundPoints[i].x;
                let y = foregroundPoints[i].y;
                
                sourceCtx.beginPath();
                sourceCtx.arc(x, y, sliderValue, 0, 2 * Math.PI);
                sourceCtx.fill();
            }
        
            sourceCtx.fillStyle = 'red';
            for (let i = 0; i < backgroundPoints.length; i++) {
                let x = backgroundPoints[i].x;
                let y = backgroundPoints[i].y;
                
                sourceCtx.beginPath();
                sourceCtx.arc(x, y, sliderValue, 0, 2 * Math.PI);
                sourceCtx.fill();
            }
        }
      },
  }
});

pxToMmInput.addEventListener('input', () => {
    if (glandLabels) {
        updateGlobalMetrics();
    }
});

toggle.addEventListener('click', () => {
    toggle.classList.toggle('active');
    pseudoScribbleEnabled = toggle.classList.contains('active');
});

function average(arr) {
        if (!arr || arr.length === 0) return 0;
        return arr.reduce((sum, v) => sum + v, 0) / arr.length;
}

function updateGlobalMetrics() {
    if (!glandLabels) {
        overallGlandNumber.textContent = "--";
        averageGlandLength.textContent = "--";
        averageGlandWidth.textContent = "--";
        AverageGlandCurvature.textContent = "--";
        averageGlandArea.textContent = "--";
        return;
    }
    const pxToMm = parseFloat(pxToMmInput.value) || 0;
    overallGlandNumber.textContent = totalGlandNum;
    averageGlandLength.textContent = (average(glandLengths) * pxToMm).toFixed(3);
    averageGlandWidth.textContent = (average(glandWidths) * pxToMm).toFixed(3);
    AverageGlandCurvature.textContent = average(glandCurvatures).toFixed(3);
    averageGlandArea.textContent = (average(glandAreas) * pxToMm * pxToMm).toFixed(3);
}

async function saveExcelWithPicker(blob) {
    // 检查浏览器支持
    if (!window.showSaveFilePicker) {
        alert("Your browser does not support file picker API. The file will be downloaded automatically.");
        fallbackDownload(blob);
        return;
    }

    // 文件保存配置
    const options = {
        suggestedName: "患者睑板腺分析报告.xlsx",
        types: [
            {
                description: "Excel File",
                accept: { "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"] },
            },
        ],
    };

    try {
        // 弹出保存对话框
        const handle = await showSaveFilePicker(options);
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
    } catch (err) {
        console.error("Save canceled or failed:", err);
        alert('File Lock!');
    }
}

function fallbackDownload(blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "患者睑板腺分析报告.xlsx";
    link.click();
    URL.revokeObjectURL(url);
}

async function saveCanvasAsPNG() {
    // 1) 获取 PNG Blob
    const blob = await new Promise((resolve) => {
        resultCanvas.toBlob(resolve, "image/png");
    });

    // 2) 如果浏览器支持 File System Access API → 让用户选择保存路径
    if (window.showSaveFilePicker) {
        try {
            const handle = await showSaveFilePicker({
                suggestedName: "segmentation_mask.png",
                types: [
                    {
                        description: "PNG image",
                        accept: { "image/png": [".png"] },
                    },
                ],
            });

            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
        } catch (err) {
            console.warn("User canceled or error:", err);
        }
    } else {
        // 3) 浏览器不支持 → fallback 默认下载
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "segmentation_mask.png";
        link.click();
        URL.revokeObjectURL(url);
    }
}

