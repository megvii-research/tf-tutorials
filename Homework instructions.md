The turns should be produced by `diff -u` commands.

```
zsc@suspicious-heyrovsky /unsullied/sharefs/zsc/isilon-share/zsc-awesome-neupeak/tutorial/config
 % diff -u 03-SVHN.base.auto_encoder 03-SVHN.base.auto_encoder.tanh
diff -u 03-SVHN.base.auto_encoder/model.py 03-SVHN.base.auto_encoder.tanh/model.py
--- 03-SVHN.base.auto_encoder/model.py  2019-02-14 09:43:57.260918707 +0800
+++ 03-SVHN.base.auto_encoder.tanh/model.py     2019-02-14 09:46:38.593203497 +0800
@@ -103,6 +103,7 @@
 
     x = O.unpool2d(x, window=2)
     x = conv2d('conv1t', x, output_nr_channel=3)
+    x = O.tanh(x)
 
     pred = x
 
Common subdirectories: 03-SVHN.base.auto_encoder/__pycache__ and 03-SVHN.base.auto_encoder.tanh/__pycache__
Common subdirectories: 03-SVHN.base.auto_encoder/train_log and 03-SVHN.base.auto_encoder.tanh/train_log
```
