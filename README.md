<h4>Prostate cANcer graDe Assessment (PANDA) Challenge</h4>
<ol>
<li> Prostate biopsy images dataset from Radboud University Medical Center and Karolinska Institute submitted to Kaggle
</li>
<li>Images from Radboud are graded with Gleason System and those from Karolinska with cancer (yes/no) and background</li>
<li>Images are multilevel tif with compressed jpeg and require finesse to extract </li>
<li>Once images are obtained, they are input into OpenCV cv2 and manipulated into 224 x 224 pixel tiles for training with ResNet50 model</li>
<li>Image preprocessing includes thresholding white background to black, normalizing the image and then transposing it to channels first for processing wiht PyTorch CNN's in ResNet50</li>
<li>Labels are reduced to 0=background, 1=healthy tissue (or stroma) and 2=cancerous so that Karolinska and Radboud can be trained together</li>
<li>One Hot Encoding is used with Binary CrossEntropy loss for training</li>
<li>Cancer is predicted on a per tile basis. Some examples are shown below</li>
<li>As noted in the dataset, the Radboud data is higher quality and the cancerous regions are much better labelled. The Karolinska predictions tend to pick up the cancer containing tiles but also tend to predict cancer in non-cancerous labelled tiles. More consistent and a larger dataset is needed to further the accuracy of the model. However, detecting the cancerous tiles and over-predicting non-cancerous tiles is preferable to not detecting the cancerous tile since it would alert the medical caregivers of the problem.</li>
<li></li>
<li></li>
<li></li>
</ol>

![Radboud Sample](https://github.com/msb1/panda/blob/master/samples/R67799.png)

![Karolinska Sample](https://github.com/msb1/panda/blob/master/samples/K9d0af.png)
