POS (Part-of-Speech) Tagging
==============================
词性标注

Train your model
--------------------
自己训练模型

###POS model
#### Folder Structure
```shell
/deepnlp
./pos
..pos_model.py
..reader.py
../data
.../en
....train.txt
....dev.txt
....test.txt
.../zh
....train.txt
....dev.txt
....test.txt
../ckpt
.../en
.../zh
```
#### Prepare corpus
First, prepare your corpus and split into 3 files: 'train.txt', 'dev.txt', 'test.txt'.
Each line in the file represents one annotated sentence, in this format: "word1/tag1 word2/tag2 ...", separated by white space.

```python
#train.txt
#English:
POS/NN tagging/NN is/VBZ now/RB done/VBN in/IN the/DT context/NN of/IN computational/JJ linguistics/NNS ./.

#Chinese:
充满/v  希望/n  的/u  新/a  世纪/n  ——/w  一九九八年/t  新年/t  讲话/n  （/w  附/v  图片/n  １/m  张/q  ）/w  
```

#### Specifying data_path
So model can find training data files. Download the source of package and put all three corpus files in the folder ../deepnlp/pos/data/zh
for your specific language option, create subfolders .../data/'your_language_code' and .../ckpt/'your_language_code'
you can change data_path setting in reader.py and pos_model.py

#### Running script
```python
python pos_model.py en # LSTM model English

python pos_model.py zh # LSTM model Chinese

python pos_model_bilstm.py en # Bi-LSTM model English

python pos_model_bilstm.py zh # Bi-LSTM model Chinese

```
#### Trained model can be found under folder ../deepnlp/pos/ckpt/'your_language_code'

