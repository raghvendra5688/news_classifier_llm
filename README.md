# News Classifier using Fine-tuned Google-bert-uncased and Modern Bert

The models were fine-tuned using 210k huffington post articles into 42 categories.

The models can handle a maximum token length of 256

The models were fine-tuned using 60 / 20 / 20 stratified splits for training, eval and test sets

The Google Bert and Modern Bert model achieves a performance of: </br>
*eval_loss*: 1.719, 1.968 </br>
*eval_f1*: 0.776, 0.786 </br>
*epoch*: 10.0

The model performance based on weighted F1-score across 42 categories are: </br>
Weighted F1 Score on Test Set: 0.7009, 0.7126

Examples Predictions:
<p> Text: Manage Your Expectations, Lower Your Stress Your outlook is a product of your own relationship with expectation. What will or won't happen, no one knows. And how we deal with the stress of not knowing, whether to hope for the best or expect the worst, the idea that our expectations always directly affect an outcome is little more than magical thinking. </p>
Predicted label: WELLNESS, Score: 0.7709, 0.8339 </br>
<p>Text: Donald Trump Tells Drought-Plagued Californians: 'There Is No Drought' "If I win, believe me, we’re going to start opening up the water." </p>
Predicted label: POLITICS, Score: 0.9552, 0.8151 </br>
<p>Text: Sticker Shock: 5 Of The World's Most Expensive Hotel Suites (PHOTOS) You can only imagine our heart palpitations when we see the nightly rates of these hotels' top-of-the-line suites.</p>
Predicted label: TRAVEL, Score: 0.9818, 0.9901 </br>

Here the results indicated are <google-bert>,<modern-bert> respectively.

