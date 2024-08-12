
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polarimetric SAR Calibration</title>
</head>
<body>

<h1>Polarimetric SAR Calibration</h1>

<h2>Overview</h2>
<p>This Python script is designed for polarimetric Synthetic Aperture Radar (SAR) calibration using corner reflector data. It processes SAR data, specifically for detecting corner reflectors, by utilizing various input files including Corner Reflector Data in <code>.csv</code> format, LLH file, and LKV file. The script primarily operates with data obtained from the UAVSAR project and includes functions to download, process, and generate corner reflector detection images.</p>

<h2>Prerequisites</h2>

<h3>Libraries Required</h3>
<ul>
    <li><code>os</code></li>
    <li><code>wget</code></li>
    <li><code>datetime</code></li>
    <li><code>requests</code></li>
    <li><code>struct</code></li>
    <li><code>tqdm</code></li>
    <li><code>numpy</code></li>
    <li><code>pandas</code></li>
    <li><code>uavsar_pytools.incidence_angle</code></li>
    <li><code>bs4</code> (BeautifulSoup)</li>
</ul>
<p>These libraries can be installed using <code>pip</code>:</p>

<pre><code>pip install wget requests tqdm numpy pandas uavsar_pytools beautifulsoup4</code></pre>

<h3>Input Files</h3>
<ul>
    <li><strong>Corner Reflector Data (.csv)</strong></li>
    <li><strong>LLH File:</strong> A file containing Latitude, Longitude, and Height information.</li>
    <li><strong>LKV File:</strong> A file containing Look Vector information.</li>
</ul>

<h2>Output</h2>
<p>The output of this script will be a Corner Reflector Detected Image.</p>

</body>
</html>
