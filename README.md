# ⚙️ kernel-anvil - Faster GPU kernels for AMD

[![Download kernel-anvil](https://img.shields.io/badge/Download%20kernel--anvil-blue?style=for-the-badge&logo=github)](https://github.com/kacieempiric822/kernel-anvil)

## 🚀 What kernel-anvil does

kernel-anvil helps speed up GPU kernels on AMD RDNA3 cards. It tunes llama.cpp MMVQ kernels for the shape of each model. That can improve decode speed on supported hardware.

Use it if you want better model speed on an AMD 7900 XTX or a similar RDNA3 GPU.

## 💾 Download

Visit this page to download and run the app on Windows:

[Download kernel-anvil](https://github.com/kacieempiric822/kernel-anvil)

Open the page in your browser, then look for the latest release or download file. Save it to your PC, then run it after the download finishes.

## 🖥️ What you need

- Windows 10 or Windows 11
- An AMD RDNA3 GPU
- Current AMD graphics drivers
- Enough free disk space for the app and model files
- llama.cpp models that use MMVQ kernels

For best results, use a high-end AMD card with plenty of VRAM. The tool is built for local model runs and GPU tuning.

## 🛠️ How to install

1. Open the download page:
   [https://github.com/kacieempiric822/kernel-anvil](https://github.com/kacieempiric822/kernel-anvil)

2. Find the latest release or app file.

3. Download the Windows file to your computer.

4. If the download is a ZIP file, extract it to a folder you can find later.

5. If the download is an EXE file, double-click it to start the app.

6. If Windows asks for permission, choose Yes.

7. Keep the app in a folder with write access so it can save tuned kernel files.

## 🧭 First run

When you start kernel-anvil for the first time, it will look for your GPU and model setup.

Follow the on-screen steps to:

- pick your AMD GPU
- point the app to your llama.cpp model files
- choose the model shape you want to tune
- start the tuning run

Let the process finish before you close the app. The first run may take a while because it builds a better kernel fit for your model.

## ⚡ How to use it

1. Start kernel-anvil.
2. Load the model shape you plan to run in llama.cpp.
3. Choose the target GPU.
4. Begin the tuning pass.
5. Save the tuned output.
6. Use the tuned kernel files with your local model run.

If you tune more than one model shape, repeat the same steps for each one. The app builds a fit for each shape, so each tuned result can help a different model layout.

## 📁 Expected folder setup

A simple folder layout can help keep things clear:

- `kernel-anvil\` — the app files
- `models\` — your llama.cpp models
- `tuned\` — saved output from kernel-anvil
- `logs\` — run details and tuning records

You can use any folder names you want, but keeping files separate makes it easier to find the tuned results later.

## 🔍 Best results

- Use the latest AMD driver
- Close other heavy GPU apps before tuning
- Keep your model files on a fast drive
- Tune one model shape at a time
- Save each tuned result with a clear name

If your system has a 7900 XTX, start with that first. The app is designed to make the most of that class of GPU.

## 🧪 What it can improve

kernel-anvil focuses on GPU kernel work for llama.cpp MMVQ models. It can help with:

- faster decode speed
- better kernel fit for a model shape
- less wasted GPU work
- more consistent runtime for local inference

Results can vary by model, driver version, and GPU setup. A tuned kernel for one model shape may not help another shape in the same way.

## 🧰 Common tasks

### Tune a new model
1. Open the app.
2. Select the new model.
3. Run tuning.
4. Save the result.
5. Use the saved output in your local workflow.

### Re-run tuning after a driver update
1. Update your AMD driver.
2. Open kernel-anvil again.
3. Load the same model.
4. Run tuning again.
5. Compare the new result with the older one.

### Move to a new PC
1. Install the same AMD driver class.
2. Copy your model files.
3. Copy your tuned output if you want to reuse it.
4. Run kernel-anvil on the new machine.
5. Tune again if the GPU setup changed.

## 🧭 Troubleshooting

### The app does not start
- Check that you downloaded the full file
- Make sure Windows finished the download
- Try running it again as admin
- Confirm that your antivirus did not block it

### The GPU is not detected
- Update your AMD driver
- Restart the PC
- Make sure your monitor is on the AMD card
- Close other GPU tools and try again

### The tuning run is slow
- This is normal for the first pass
- Use a smaller model first
- Close browser tabs and games
- Make sure no other app is using the GPU

### The output does not help much
- Tune again with the exact model shape
- Check that you used the right GPU
- Keep driver versions steady when comparing results
- Test with the same prompt and same runtime setup

## 📌 File handling tips

Keep the original download file in case you need it again.

Store tuned results in a separate folder so you can tell them apart from raw model files.

If you want to compare results, keep notes on:

- model name
- model shape
- GPU used
- driver version
- tuning date

## 🔗 Download again

[Visit the kernel-anvil download page](https://github.com/kacieempiric822/kernel-anvil)

## 🧱 Project focus

kernel-anvil is built for users who want better local model speed on AMD hardware. It centers on RDNA3 GPUs and model-specific MMVQ tuning. That makes it a good fit for users who run llama.cpp on a Windows PC and want stronger decode performance without changing their whole setup