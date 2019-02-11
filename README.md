# Sing4Me - Song lyric Generation with fastai

notebook-code.ipynb file has all the code that was used for this model. 
The learner was built with the fastai library which sits on top of Pytorch. 

This uses the language model learner which is pretrained with the wikitext_103 weights. 

The lyrics of Coldplay, Taylor Swift, Selena Gomez, Sugarland, Lorde, Lana Del Rey, Bob Dylan, Keane, Onerepublic, Snow PAtrol, Imagine Dragons, The Fray, Ed Sheeran, Shawn Mendes, Jason Myraz, and Colbie Callait were used to fine tune the language model. The dataset was grabbed from Kaggle - artimous/every-song-you-have-heard-almost. 

The web app is hosted on render.com. 

This is all thanks to btahir and the project to make a Walt Whitman novel with fastai. 
I forked the repo as a base for my web app.
![alt text](https://github.com/btahir/leaves-of-ai/)

The next step is to use beam search to optimize these results. 
