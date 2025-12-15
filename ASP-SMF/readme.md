# **ASP-SMF: Adaptive Snow-Phenology Shape Model Fitting**

**Daily Seamless 500-m Snow Depth Mapping in Northern Hemisphere Mountains via In-situ Independent Adaptive Snow-Phenology Shape Model Fitting**

This repository contains the source code and sample data for the **ASP-SMF algorithm**. ASP-SMF is a novel framework designed to generate daily, seamless, high-resolution (500 m) snow depth maps by synergizing **ERA5-Land** reanalysis, **Sentinel-1** SAR, and **IMS** data. It operates independently of in-situ observations, making it ideal for data-scarce mountain regions.

## **📂 Repository Structure**

.  
├── ASP\_SMF\_Threshold\_Determination.py  \# Script for Step 1: Calculating adaptive thresholds  
├── ASP\_SMF\_Workflow\_SingleFile.py      \# Script for Step 2: Main snow depth estimation  
├── data/  
│   ├── step1\_Threshold\_Determination/  
│   │   ├── input\_time\_series/          \# Place random point time series for Step 1 here  
│   │   └── output\_analysis/            \# Step 1 outputs (Statistics & Decision Tree Plots)  
│   └── step2\_ASP\_SMF\_Workflow/  
│       └── Sample\_timeSeries1.csv      \# Sample pixel time series for Step 2  
└── README.md

## **🚀 Usage Guide**

The workflow consists of two main steps:

1. **Threshold Determination**: Calculate regionally adaptive physical thresholds.  
2. **Snow Depth Estimation**: Generate the final daily snow depth time series.

### **Step 1: Threshold Determination**

First, run ASP\_SMF\_Threshold\_Determination.py to derive the specific snow phenology thresholds for your study area.

#### **1\. Input Data**

* **Directory**: ./data/step1\_Threshold\_Determination/input\_time\_series  
Please unzip input_time_series.zip before running Step 1.
* **Format**: CSV files containing time series data for random points within your study region.  
* **Required Columns**:  
  * Date: Format YYYYMMDD or YYYY-MM-DD.  
  * ERA5\_Land\_snow\_depth: Snow depth from ERA5-Land.  
  * ERA5\_Land\_snowfall\_sun: Daily snowfall sum (from ERA5-Land snowfall\_sum).  
  * ERA5\_Land\_snowmelt\_sun: Daily snowmelt sum (from ERA5-Land snowmelt\_sum).

**Data Source**: ERA5-Land data can be downloaded from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu) or via Google Earth Engine (GEE).

#### **2\. Running the Script**

Run the script to process the input time series:

python ASP\_SMF\_Threshold\_Determination.py

#### **3\. Output & Threshold Extraction**

The results will be saved in ./data/step1\_Threshold\_Determination/output\_analysis. You need to extract four key parameters from these outputs to configure Step 2:

* **RATE\_ACCUMULATION** & **RATE\_MELT**:  
  * Open **3\_day\_quantile\_statistics.xlsx**.  
  * Find the values in the last two rows labeled THRESHOLD\_POSITIVE and THRESHOLD\_NEGATIVE.  
* **SUBTRACT\_FALL\_MELT\_LOWER** & **SUBTRACT\_FALL\_MELT\_UPPER**:  
  * Open **Detailed\_Decision\_Boundary\_Annotated.png**.  
  * Read the decision tree split values from the annotated plot. These values define the physical boundaries for the stable snow phase based on net mass flux.

### **Step 2: Snow Depth Estimation Workflow**

After obtaining the thresholds, update the configuration in ASP\_SMF\_Workflow\_SingleFile.py and run the estimation.

#### **1\. Configuration**

Open ASP\_SMF\_Workflow\_SingleFile.py and update the THRESHOLDS dictionary with the values obtained from Step 1:

THRESHOLDS \= {  
    'RATE\_ACCUMULATION': 0.03396963,       \# Update from Step 1 Output  
    'RATE\_MELT': \-0.00793871,              \# Update from Step 1 Output  
    'SUBTRACT\_FALL\_MELT\_LOWER': \-0.0028,   \# Update from Step 1 Output  
    'SUBTRACT\_FALL\_MELT\_UPPER': 0.0078,   \# Update from Step 1 Output  
    'SUM\_FALL\_MELT': 0.0000,  \# Update from Step 1 Output  
    'SNOW\_DEPTH\_STABLE': 0.02,  
}

#### **2\. Input Data**

* **Directory**: ./data/step2\_ASP\_SMF\_Workflow/  
* **File**: Sample\_timeSeries1.csv (or your own pixel data).  
* **Required Columns**:  
  * Date: Date of observation.  
  * ERA5\_Land\_snow\_depth, ERA5\_Land\_snowfall\_sun, ERA5\_Land\_snowmelt\_sun: (Same as Step 1).  
  * **IMS**: Interactive Multisensor Snow and Ice Mapping System data.  
  * **S1\_dry**: Sentinel-1 dry snow depth.

**Data Sources & Preprocessing**:

* **IMS**: Downloaded from [NOAA at NSIDC DAAC](https://www.google.com/search?q=https://nsidc.org/data/g02156).  
  * *Value Definitions*: 0: Outside Hemisphere, 1: Open Water, 2: Snow Free Land, 3: Sea/Lake Ice, 4: Snow Covered.  
  * *Usage*: We exclusively use **Value 2 (Snow Free Land)** to constrain the model with Snow Depth \= 0\.  
* **S1\_dry**: Downloaded from the [C-SNOW project data portal](https://ees.kuleuven.be/project/c-snow).  
  * *Usage*: Ensure the provided wet snow flag is applied to remove wet snow pixels. Only **dry snow** retrievals should be retained in the S1\_dry column.

#### **3\. Running the Script**

python ASP\_SMF\_Workflow\_SingleFile.py

#### **4\. Outputs**

The script generates two files in the same directory as the input:

1. **{filename}\_Optimized.csv**:  
   * Contains the full processed time series.  
   * The final result is in the last column: **Optimized\_Snow\_Depth**.  
2. **{filename}\_Optimized\_Plot.png**:  
   * A visualization comparing the original ERA5-Land input, Sentinel-1 anchors, and the final optimized snow depth.

*(Example visualization of the ASP-SMF output)*

## **🌍 Regional Application**

To generate a snow depth map for an entire region:

1. Prepare the time series data for every pixel in your region of interest (aligned to the 500 m grid).  
2. Apply **Step 1** once to determine the regional thresholds.  
3. Execute **Step 2** for all pixels. Due to the independent nature of the pixel-wise processing, this step can be easily parallelized using batch processing or multiprocessing frameworks.

## **📄 Reference**

If you use this code, please cite the following paper:

> **Cheng Zhang, Lingmei Jiang, et al.** (2025). Daily Seamless 500-m Snow Depth Mapping in Northern Hemisphere Mountains via In-situ Independent Adaptive Snow-Phenology Shape Model Fitting. *(Under Review)*.


**License**: MIT