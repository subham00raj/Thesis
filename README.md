<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
<h1>Polarimetric SAR Calibration</h1>
<h2>Overview</h2>
<p>This Python script is designed for polarimetric Synthetic Aperture Radar (SAR) calibration using corner reflector data. It processes SAR data, specifically for detecting corner reflectors, by utilizing various input files including Corner Reflector Data in <code>.csv</code> format, LLH file, and LKV file. The script primarily operates with data obtained from the UAVSAR project and includes functions to download, process, and generate corner reflector detection images.</p>
<h2>Prerequisites</h2>
<h3>Libraries Required</h3>

<p>These libraries can be installed using <code>pip</code>:</p>
<pre><code>pip install -r requirements.txt</code></pre>
<h3>Input Files</h3>
These files can be downloaded using <code> get_uavsar_data </code> and <code> get_corner_reflector_data </code> function. This fetches the following files
<ul>
    <li><strong>SLC File:</strong> UAVSAR quad pol image data in complex binary format.</li>
    <li><strong>LLH File:</strong> A file containing Latitude, Longitude, and Height information.</li>
    <li><strong>LKV File:</strong> A file containing Look Vector information.</li>
    <li><strong>CSV File:</strong> A file containing detailed information of each corner reflector in the image.</li>
</ul>
<h2>Output</h2>
<p>The output of this script will be a Corner Reflector Detected Image.</p>
</body>
</html>
