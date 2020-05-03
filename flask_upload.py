from flask import Flask, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np # linear algebra
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import shutil 


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

ALLOWED_EXTENSIONS=set(['jpg', 'jpeg'])  #ALLOWED FILE EXTENTION 

def allowed_file(filename):
    return '.' in filename and filename.lower().rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)


model = torch.load("./static/model_gorlo.h5")

UPLOAD_FOLDER = './uploads/unknown'
data_root = './uploads'
test_dir = './uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_prediction():
     
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])  
    test_dataset = ImageFolderWithPaths(test_dir, val_transforms)
    #test_dataset = torchvision.datasets.ImageFolder(test_dir, val_transforms)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0)
    model.eval()

    test_predictions = []
    test_img_paths = []
    for inputs, labels, paths in tqdm(test_dataloader):
        #inputs = inputs.to(device)
        #labels = labels.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())
        test_img_paths.extend(paths)
    
    test_predictions = np.concatenate(test_predictions)

    inputs, labels, paths = next(iter(test_dataloader))
    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'red' if pred > 0.5 else 'normal')
    submission_df['id'] = submission_df['id'].str.replace('uploads/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')    
    return submission_df


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #img = Image.open(io.BytesIO(file))
            res = get_prediction()
            print(res)
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <head>
    <title> STOP CORONOVID SERVER</title>
    </head>
    <body>
    <h1 align=center> STOP CORONOVID19 project by DataStalker:</h1>
    <p align=center><img src="\static\stop_p.png"
        alt="Town trip"></p>
    <p> Chek foto of throat for Coronovid-19 and over disease.
    Server detect flue.</p>    
    <title>Upload new File</title>
    <table>
   <tr>
    <th>
    <p>Upload new File</p>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>

    </th>
    <th>
        <p>COVID-19 is an viral respiratory infections agent that shows pronounced intoxication of the body and problems with the respiratory and digestive systems</p></th>
   </tr>
   <tr>
    <td>
        <p>COVID-19 spreads:
Air-drip in case of sneezing and coughing;
In the contact way
<p>What to do if you have COVID-19 symptoms:</p>
<p>see a doctor;
Not to self-medicate</p> </p>

</td>
    <td>
        <p><p>How to prevent COVID-19:</p>
<p>Do not go to the focus of the disease;</p>
<p>Avoid attending mass events;</p>
<p>CAREFULLY WASH HANDS;</p>
<p>Use medical masks;</p>
<p>Avoid close contact with people who have symptoms of the disease.</p> </p>
</td>
  </tr>
 </table>
    </body>
    </html>
    '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(UPLOAD_FOLDER)
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

#from werkzeug import SharedDataMiddleware
#app.add_url_rule('/uploads/<filename>', 'uploaded_file',
#                 build_only=True)
#app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
#    '/uploads':  app.config['UPLOAD_FOLDER']
#})

if __name__ == '__main__':
    app.run(debug=True)