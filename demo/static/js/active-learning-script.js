const datasetUpload = document.getElementById("datasetUpload");
const datasetName = document.getElementById("datasetName");
const modelUpload = document.getElementById("modelUpload");
const modelName = document.getElementById("modelName");
const runBtn = document.getElementById("runActiveLearningBtn");
const samplingStrategy  = document.getElementById("samplingStrategy");
const budgetCount       = document.getElementById("budgetCount");
const statusMessage     = document.getElementById("statusMessage");
const labeledDatasetUpload = document.getElementById("labeledDatasetUpload");
const labeledDatasetName = document.getElementById("labeledDatasetName");
const folderNameEl = document.getElementById("folderName");
const chooseFolderBtn = document.getElementById("chooseFolderBtn");
const labeledDatasetGroup = labeledDatasetUpload.closest(".form-group");
let unlabeledDataOK = false;
let labeledDataOK = false;
let modelOK = false;

let unlabeledDataList = [];   // 保存 base64
let labeledDataList = [];
let dirHandle = null;
samplingStrategy.addEventListener("change", () => {
    const val = samplingStrategy.value;
    const needLabeled = ["PHS", "SDS", "Core-set"].includes(val);
    labeledDatasetGroup.style.display = needLabeled ? "block" : "none";
    checkReady();
});


datasetUpload.addEventListener("change", async (e) => {
    datasetName.classList.remove("selected");
    const files = Array.from(e.target.files);

    unlabeledDataList = []; // 仍然保留，只是存 File，而不是 base64

    if (!files || files.length === 0) {
        datasetName.innerText = "未选择任何文件/文件夹";
        unlabeledDataOK = false;
        checkReady();
        return;
    }

    // 过滤 png/jpg
    const imageFiles = files.filter(file => {
        const lower = file.name.toLowerCase();
        return lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg");
    });

    if (imageFiles.length === 0) {
        datasetName.innerText = "未找到图片文件 (.png / .jpg)";
        unlabeledDataOK = false;
        checkReady();
        return;
    }

    datasetName.innerText = "数据上传中";

    unlabeledDataList = imageFiles.map(f => ({
        name: f.name,
        file: f               // ✔ 保存 File，不是 base64
    }));

    // === 上传给后端（FormData）===
    const formData = new FormData();
    for (let f of imageFiles) {
        formData.append("files", f, f.name);
    }

    await fetch("/upload_unlabeled_data", {
        method: "POST",
        body: formData
    });

    unlabeledDataOK = true;
    checkReady();
    datasetName.innerText = `已选择 ${imageFiles.length} 张图像`;
    datasetName.classList.add("selected");
});


labeledDatasetUpload.addEventListener("change", async (e) => {
    labeledDatasetName.classList.remove("selected");
    const files = Array.from(e.target.files);
    labeledDataList = [];  // 仍然存 File，不存 base64

    if (!files || files.length === 0) {
        labeledDatasetName.innerText = "未选择任何文件/文件夹";
        labeledDataOK = false;
        checkReady();
        return;
    }

    // 过滤 png/jpg
    const imageFiles = files.filter(f => {
        const lower = f.name.toLowerCase();
        return lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg");
    });

    if (imageFiles.length === 0) {
        labeledDatasetName.innerText = "未找到图片文件 (.png / .jpg)";
        labeledDataOK = false;
        checkReady();
        return;
    }

    labeledDatasetName.innerText = "数据上传中";

    labeledDataList = imageFiles.map(f => ({
        name: f.name,
        file: f
    }));

    // === 用 FormData 上传到后端===
    const formData = new FormData();
    for (let f of imageFiles) {
        formData.append("files", f, f.name);
    }

    await fetch("/upload_labeled_data", {
        method: "POST",
        body: formData
    });

    console.log("已标注数据上传完成:", labeledDataList.length);

    labeledDatasetName.innerText = `已选择 ${imageFiles.length} 张图像`;
    labeledDatasetName.classList.add("selected");

    labeledDataOK = true;
    checkReady();
});


modelUpload.addEventListener("change", async (e) => {
    modelName.classList.remove("selected");
    const file = e.target.files[0];
    if (!file || file.length === 0) {
        modelName.innerText = "未选择模型文件";
        modelOK = false;
        checkReady();
        return;
    }

    modelName.innerText = "模型上传中";

    // 使用 FormData 上传文件
    const formData = new FormData();
    formData.append("model_file", file);

    try {
        const res = await fetch("/upload_model", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        alert(data.msg);
        modelName.innerText = file.name;
        modelName.classList.add("selected");
        modelOK = true;
        checkReady();
    } catch (err) {
        console.error(err);
        alert("上传失败");
    }
});

chooseFolderBtn.addEventListener("click", async () => {
    folderNameEl.classList.remove("selected");
    try {
        dirHandle = await window.showDirectoryPicker();
        folderNameEl.innerText = `已选择文件夹：${dirHandle.name}`;
        folderNameEl.classList.add("selected");
    } catch (err) {
        console.warn("用户取消了文件夹选择：", err);
        folderName.innerText = "未选择文件夹";
        dirHandle = null;
    }
});

runBtn.addEventListener("click", async () => {
    try {
        // ✅ 0. 检查用户是否选择了文件夹
        if (!dirHandle) {
            alert("请先选择保存文件夹！");
            return;
        }
         // ✅ 确认权限仍有效
        let perm = await dirHandle.queryPermission({ mode: "readwrite" });
        if (perm !== "granted") {
            perm = await dirHandle.requestPermission({ mode: "readwrite" });
            if (perm !== "granted") {
                statusMessage.innerText = "⚠️ 没有访问该文件夹的权限，请重新选择！";
                dirHandle = null;
                return;
            }
        }

        // ✅ 1. 获取用户输入参数
        const strategy = samplingStrategy.value;
        const budget = Number(budgetCount.value);
        statusMessage.innerText = "正在筛选最有标注价值的样本...";

        const payload = { strategy, budget };

        // ✅ 2. 调用后端运行主动学习逻辑
        const res = await fetch("/run_active_learning", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const result = await res.json();

        // ✅ 3. 检查是否有选中样本
        if (!result.selected_names || result.selected_names.length === 0) {
            statusMessage.innerText = "未选中样本！";
            return;
        }

        statusMessage.innerText = `选中 ${result.selected_names.length} 张图像，准备保存...`;

        // ✅ 4. 匹配需要保存的 File 对象
        const selectedImages = unlabeledDataList.filter(item =>
            result.selected_names.includes(item.name)
        );

        if (selectedImages.length === 0) {
            alert("未在本地缓存中找到对应图片！");
            return;
        }

        // ✅ 5. 将每张 base64 图片写入选定文件夹
        for (let img of selectedImages) {
            const fileHandle = await dirHandle.getFileHandle(img.name, { create: true });
            const writable = await fileHandle.createWritable();

            // const blob = base64ToBlob(img.base64);
            await writable.write(img.file);
            await writable.close();
        }

        // ✅ 6. 保存成功提示
        statusMessage.innerText = `✅ 已成功保存 ${selectedImages.length} 张最有标注价值的图像到文件夹 "${dirHandle.name}"！`;

    } catch (err) {
        if (err.name === "AbortError") {
            statusMessage.innerText = "用户已取消文件夹选择。";
        } else if (err.name === "SecurityError") {
            statusMessage.innerText = "⚠️ 没有访问该文件夹的权限，请重新选择！";
            dirHandle = null;
        } else {
            console.error("主动学习失败：", err);
            statusMessage.innerText = "❌ 主动学习失败，请检查控制台。";
        }
    }
});

// 将 File 转为 Base64
function fileToBase64Async(file) {
    return new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function checkReady() {
    statusMessage.innerText = "";
    const val = samplingStrategy.value;
    const needLabeled = ["PHS", "SDS", "Core-set"].includes(val);
    const ready = unlabeledDataOK && modelOK && (!needLabeled || labeledDataOK);
    runBtn.disabled = !ready;
}

function base64ToBlob(base64) {
    const byteString = atob(base64.split(",")[1]);
    const mimeString = base64.split(",")[0].split(":")[1].split(";")[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
}

