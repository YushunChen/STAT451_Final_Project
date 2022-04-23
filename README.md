# STAT 451 Project Proposal: Anti-Phishing Using Machine Learning

Group member: Oliver Chen (ychen939), Yizhou Chen (ychen884), Sirui Chu (schu46), Weiyu Xu (wxu232)

## Introduction
Phishing is a common form of cyber attacks, where 80% of reported security incidents in 2021 are phishing attacks. According to CISCO's 2021 Cybersecurity Threat Trends report, approximiately 90% of data breaches on the internet came from phishing ([article](https://spanning.com/blog/cyberattacks-2021-phishing-ransomware-data-breach-statistics/#:~:text=How%20common%20was%20phishing%20in,breaches%20occur%20due%20to%20phishing.)). Phishing attackes may take the forms of emails, texts, or phone calls during which the victim is asked to enter sensitive information such as credit card numbers and login credentials. In order to contribute to the anti-phishing efforts, the goal of this project is to identify phishing websites given their URLs using machine learning methods.

## Read Data
The data set is from Kaggle ([data set link](https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning?resource=download)). This dataset consists of 5000 phishing websites and 5000 legitimate websites from January to May 2015 and from May to June 2017. There are 48 features in total, which will be explained later.


```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```


```python
df = pd.read_csv('./data.csv')
df[:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NumDots</th>
      <th>SubdomainLevel</th>
      <th>PathLevel</th>
      <th>UrlLength</th>
      <th>NumDash</th>
      <th>NumDashInHostname</th>
      <th>AtSymbol</th>
      <th>TildeSymbol</th>
      <th>NumUnderscore</th>
      <th>...</th>
      <th>IframeOrFrame</th>
      <th>MissingTitle</th>
      <th>ImagesOnlyInForm</th>
      <th>SubdomainLevelRT</th>
      <th>UrlLengthRT</th>
      <th>PctExtResourceUrlsRT</th>
      <th>AbnormalExtFormActionR</th>
      <th>ExtMetaScriptLinkRT</th>
      <th>PctExtNullSelfRedirectHyperlinksRT</th>
      <th>CLASS_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>72</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>144</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>58</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>79</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



### Description of Variables
Our group has written an Excel file containing all the 48 variables and their detailed information. The original data has 50 columns. The remaining 2 columns are the id/index column and the CLASS_LABEL column where 1 stands for phishing websites and 0 stands for legitimate websites. We converted the Excel file to a markdown table for better presentation.

| Variable                           | Value type    | Description                                                                                                                                                                  |
| ---------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| id                                 | Discrete(int) | the index of the URL                                                                                                                                                         |
| NumDots                            | Discrete(int) | the number of dots in the URL                                                                                                                                                |
| SubdomainLevel                     | Discrete(int) | the level of subdomain in the URL                                                                                                                                            |
| PathLevel                          | Discrete(int) | the depth of the path in the URL                                                                                                                                             |
| UrlLength                          | Discrete(int) | the total characters in the URL                                                                                                                                              |
| NumDash                            | Discrete(int) | the number of “-” in URL                                                                                                                                                     |
| NumDashInHostname                  | Discrete(int) | the number of “-” in hostname part of URL                                                                                                                                    |
| AtSymbol                           | Binary        | if “@” symbol exist in URL                                                                                                                                                   |
| TildeSymbol                        | Binary        | if “ ∼ ” symbol exist in URL                                                                                                                                                 |
| NumUnderscore                      | Discrete(int) | the number of "\_" in URL                                                                                                                                                    |
| NumPercent                         | Discrete(int) | the number of “%” in URL                                                                                                                                                     |
| NumQueryComponents                 | Discrete(int) | the number of query parts in URL                                                                                                                                             |
| NumAmpersand                       | Discrete(int) | the number of “&” in URL                                                                                                                                                     |
| NumHash                            | Discrete(int) | the number of “#” in URL                                                                                                                                                     |
| NumNumericChars                    | Discrete(int) | the number of numeric characters in the URL                                                                                                                                  |
| NoHttps                            | Binary        | if HTTPS exist in URL                                                                                                                                                        |
| RandomString                       | Binary        | if random strings exist in URL                                                                                                                                               |
| IpAddress                          | Binary        | if IP address is used in hostname part of URL                                                                                                                                |
| DomainInSubdomains                 | Binary        | if TLD or ccTLD is used as part of subdomain in URL                                                                                                                          |
| DomainInPaths                      | Binary        | if TLD or ccTLD is used in the path of URL                                                                                                                                   |
| HttpsInHostname                    | Binary        | if HTTPS in obfuscated in hostname part of URL                                                                                                                               |
| HostnameLength                     | Discrete(int) | the total characters in hostname part of URL                                                                                                                                 |
| PathLength                         | Discrete(int) | the total characters in path of URL                                                                                                                                          |
| QueryLength                        | Discrete(int) | the total characters in query part of URL                                                                                                                                    |
| DoubleSlashInPath                  | Binary        | if “//” exist in the path of URL                                                                                                                                             |
| NumSensitiveWords                  | Discrete(int) | the number of sensitive words (i.e., “secure”, “account”, “webscr”, “login”, “ebayisapi”, “signin”, “banking”, “confirm”) in webpage URL                                     |
| EmbeddedBrandName                  | Binary        | if most frequent domain name in the HTML content appears in subdomain or path of URL                                                                                         |
| PctExtHyperlinks                   | Continuous    | the percentage of external hyperlinks in webpage HTML source code                                                                                                            |
| PctExtResourceUrls                 | Continuous    | the percentage of external resource URLs in webpage HTML source code                                                                                                         |
| ExtFavicon                         | Binary        | if favorite icon/shortcut icon/URL icon from a domian name that is different from the webpage URL domain name                                                                |
| InsecureForms                      | Binary        | if the form action attribute has a URL without HTTPS                                                                                                                         |
| RelativeFormAction                 | Binary        | if the form action attribute has a relative URL                                                                                                                              |
| ExtFormAction                      | Binary        | if the form action attribute has a URL from an external domain                                                                                                               |
| AbnormalFormAction                 | Categorical   | if the form action attribute has any of the following abnormal fields: "#", "about:blank", "", or "javascript:true"                                                          |
| PctNullSelfRedirectHyperlinks      | Continuous    | the percentage of hyperlinks fields containing empty value, self-redirect value such as “#”, the URL of current webpage, or some abnormal value such as “file://E:/”         |
| FrequentDomainNameMismatch         | Binary        | if the most frequent domain name in HTML source code does not match the webpage URL domain name                                                                              |
| FakeLinkInStatusBar                | Binary        | if HTML source code has JavaScript command onMouseOver to display a fake URL in the status bar                                                                               |
| RightClickDisabled                 | Binary        | if HTML source code contains JavaScript command to disable right click function                                                                                              |
| PopUpWindow                        | Binary        | if HTML source code contains JavaScript command to launch pop-ups                                                                                                            |
| SubmitInfoToEmail                  | Binary        | if HTML source code contains the HTML "mailto" function                                                                                                                      |
| IframeOrFrame                      | Binary        | if iframe or frame is used in HTML source code                                                                                                                               |
| MissingTitle                       | Binary        | if the title tag is empty in HTML source code                                                                                                                                |
| ImagesOnlyInForm                   | Binary        | if the form scope in HTML source code contains no text at all but images only                                                                                                |
| SubdomainLevelRT                   | Categorical   | the number of dots in hostname part of webpage URL, and values are generated through thresholds.                                                                             |
| UrlLengthRT                        | Categorical   | the total number of characters in the URL, and values are generated through thresholds.                                                                                      |
| PctExtResourceUrlsRT               | Categorical   | the percentage of external resource URLs in HTML source code, and values are generated through thresholds.                                                                   |
| AbnormalExtFormActionR             | Categorical   | if the form action attribute contains a foreign domain, “about:blank” or an<br>empty string, and values are generated through thresholds.                                    |
| ExtMetaScriptLinkRT                | Categorical   | the percentage of meta, script and link tags containing external URL in the attributes, and values are generated through thresholds.                                         |
| PctExtNullSelfRedirectHyperlinksRT | Categorical   | the percentage of hyperlinks in HTML source code that uses different domain names, starts with “#”, or using “JavaScript ::void(0)”. Values are generated through thresholds |

### Description of Question

The central question of this project is to determine whether a website is a phishing website given its webpage information (such as URL).

## Machine Learning Methods

To start with, we split the data into training and validation data. We intend to first use algorithm selection on the splitted training and validation data. These include models such as logistic regression, support vector machine, ID3 decision tree, and kNN. Based on the results, we continue to use ensemble learning, including bagging, random forest, and gradient boosting. Lastly, we determine the best method and report our results.


```python

```
