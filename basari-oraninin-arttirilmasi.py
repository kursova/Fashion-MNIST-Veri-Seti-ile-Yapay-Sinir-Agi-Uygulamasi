#gerekli kütüphaneleri yükleme
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#veri setini yükleme
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''
Fashion MNIST veri seti 10 kategoride toplamda 70.000 gri tonlamalı görüntü içerir. 
Görüntüler, (28 x 28 piksel) tek tek giyim eşyalarını göstermektedir
'''


#örnek veri (eğitim verisi)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
![image](https://www.kaggleusercontent.com/kf/168832389/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ZIAgGu5FmZjqpfSEBKJfrg.y336fifw-ixlQ2n2jJVPDfxOoC2OWW6-k_UcgBgSHwYdyVclWetQ_xYzmSdRv4fdXlMuOXvpg3PffCjrS5ru6s2JbP7MS-tmiogvcU4ooxXno0NHsfCPL5i20UBwAX-4FaoYkejwS2_4VcJREEMotSxRupCG3tcAEIwq9BAJM7fldpclMgPaBURRHCLUrYJ_1C2KcVrqd7ttpoVp7Zlvm12k3zgmnzBmH_J1KXG7vmfHJaHFzhPiGX06elhSS2rhnUeLL7sTiw7uJJweVm_Bx1AQUmCYl4l_fFzqPRtxgabb1g5seRjZwhAxviboc8X8EFRU6ib2k8QcCCh1cm0-yIJ9zEe_8VaFq1UFtKN6fXnUZ53R3_W5UcQPotgZSHbVZaS-eBvLM2CnHnEdt237XLdX48vWqaJmRFs4g2aAmv9wgweVWFXocVOfaz8f6sX6zdYpYxHVfFY3iWWPioT_y2ZQ1-x9gn0olY08-0tj25P_4qk1B3GKPs9z6pOtFO_QrhVILln6Ila5c1rWV9Qu2pPt0RFdYGbFl6QQNFh7JxuQNAubYwhDhrtG36ykoEsy_N9leOFvmG80ZrEg5DUdanriZSPI69kcoHzw2qTLqa6YPm6J21jq2Cc_ysazyNKbV1YuBqAePj3DrArNQ0q7_uWChHAeXM18TzTyMAE029IbDQ1ghwG9E-FHqFspbt6Q.qCa4-fZDBe-vIQ832BKIcg/__results___files/__results___0_0.png)
#örnek veri (test verisi)
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#normalleştirme (değeleri 0-1 arasında gösterme)
train_images = train_images / 255.0

test_images = test_images / 255.0

#modeli kurma (burada farklı fonksiyonlar denedim. En alttaki yorumlarda nasıl sonuçlar verdiklerini yazdım.)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(8, activation='elu'), #nöron sayısını değiştirdim.    
    tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dense(32, activation='gelu'), #katman ekledim
    tf.keras.layers.Dropout(0.1), #düzenlileştirme
    tf.keras.layers.Dense(64, activation='elu'), #katman ekledim
    tf.keras.layers.Dropout(0.2), #düzenlileştirme
    tf.keras.layers.Dense(10,activation="softmax")
])

#özet
model.summary()

# modeli derleyelim (compile)
model.compile(optimizer='adam', #diğer optimizasyon yöntemleri: SGD, Adadelta, Adagrad, RMSProd
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#verilerimiz 0-1 arasında değişseydi CategoricalCrossentropy kullanırdık.
              metrics=['accuracy'])

#uygulama

hist=model.fit(train_images, train_labels,validation_data=(test_images,test_labels), epochs=8) #farklı epoch numaraları denedik.

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#eğitim ve test doğrulama kaybını ve doğruluğu çıktısı
print("Eğitim Doğruluğu: ",
     hist.history['accuracy'][-1])
print("Test Doğruluğu: ",
     hist.history['val_accuracy'][-1])
print("Eğitim Kaybı: ",
     hist.history['loss'][-1])
print("Test Kaybı: ",
     hist.history['val_loss'][-1])

 #Eğitim ve dtest doğrulama kaybını ve doğruluğunu görselleştirme 
    
plt.plot(hist.history['loss'], label='Eğitim Kaybı') 
plt.plot(hist.history['val_loss'], label='Doğrulama Kaybı') 
plt.xlabel('Epoch')
plt.ylabel('Kayıp') 
plt.legend() 
plt.show() 

plt.plot(hist.history['accuracy'], label='Eğitim Doğruluğu') 
plt.plot(hist.history['val_accuracy'], label='Doğrulama Doğruluğu') 
plt.xlabel('Epoch') 
plt.ylabel('Doğruluk') 
plt.legend() 
plt.show()

'''
son durumda aşağıdaki sonuca ulaştım. Başta 0.82 olarak verilen eğitim ve test verileri arasındaki başarı değerini, yaptığım değişikliklerle, 0.86'ya yükselttim.
Eğitim Doğruluğu:   0.8628666400909424
Test Doğruluğu:     0.8468000292778015
Eğitim Kaybı:       0.37758514285087585
Test Kaybı:         0.41748255491256714
'''
#bu sonuca ulaşana kadar birçok deneme yaptım. Bu denemeleri ve aldığım sonuçları aşağıda görebilirsiniz.

#aktivasyon kodunu elu yaptığımızda değerlerimiz kötüleşti.
'''
accuracy: 0.7509 - loss: 0.7308 - val_accuracy: 0.7423 - val_loss: 0.7426

leaky relu kullanınca : accuracy: 0.8072 - loss: 0.5833 - val_accuracy: 0.7955 - val_loss: 0.6080
relu olarak değiştirdiğimizde ve dropout kullanarak düzenlileştirme yaptığımızda accuracy: 0.8443  Test accuracy: 0.844299 loss: 0.4480

nöronlardan biri 16 olarak değiştirildi. yeni katman eklendi. dropout eklendi. epoch 15 olarak değiştirildi.
accuracy: 0.8809 - loss: 0.3292 - val_accuracy: 0.8574 - val_loss: 0.4006
'''

#drop out kullanımı
'''
drop out 0.2:  accuracy: 0.8242 - loss: 0.4896 - val_accuracy: 0.8376 - val_loss: 0.457
drop out 0.3: accuracy: 0.7916 - loss: 0.5778 - val_accuracy: 0.7974 - val_loss: 0.5574
drop out 0.4: accuracy: 0.7584 - loss: 0.6606 - val_accuracy: 0.8134 - val_loss: 0.5313               
drop out 0.1: accuracy: 0.8468 - loss: 0.4271 - val_accuracy: 0.8441 - val_loss: 0.4336
drop out fonksiyonunda en iyi sonucu 0.1 değeri verdi. bunu kullanabiliriz.
'''
#katman sayılarını arttırıp azaltalım
'''
8-16: Eğitim Doğruluğu:         0.8410000205039978 Test Doğruluğu:  0.842199981212616 Eğitim Kaybı:  0.4414486587047577 Test Kaybı:  0.4367920458316803
8-16-32: Eğitim Doğruluğu:      0.8458499908447266 Test Doğruluğu:  0.843999981880188 Eğitim Kaybı:  0.42673638463020325 Test Kaybı: 0.43116965889930725
8-16-32-64: Eğitim Doğruluğu:   0.8468999862670898 Test Doğruluğu:  0.848800003528595 Eğitim Kaybı:  0.4220840632915497 Test Kaybı:  0.42227157950401306
'''
# aktivasyon fonksiyonlarını değiştirelim
'''
leaklyrelu: Eğitim Doğruluğu:   0.8445333242416382 Test Doğruluğu:  0.8450999855995178 Eğitim Kaybı:  0.42953500151634216 Test Kaybı:  0.4341968894004822
Swih: Eğitim Doğruluğu:         0.839900016784668 Test Doğruluğu:   0.8389999866485596 Eğitim Kaybı:  0.43628203868865967 Test Kaybı:  0.4351336658000946
'''
#diğer denemeler
#drop out son katmana eklendi:(overfitting var gibi) 
'''
Eğitim Doğruluğu:   0.8705166578292847 
Test Doğruluğu:     0.8529999852180481 
Eğitim Kaybı:       0.35146471858024597 
Test Kaybı:         0.4062526822090149
'''
#drop out 0.2 yaptım: 
'''
Eğitim Doğruluğu:   0.8673333525657654 
Test Doğruluğu:     0.8460000157356262 
Eğitim Kaybı:       0.36401379108428955 
Test Kaybı:         0.4285202622413635
'''
#gelu kullandığımızda: 
'''
Eğitim Doğruluğu:   0.8684999942779541 
Test Doğruluğu:     0.8529000282287598 
Eğitim Kaybı:       0.356443852186203 
Test Kaybı:         0.3989572525024414
'''
#epoch değerinini arttırdığımızda overfitting oluştu: 
''' 
Eğitim Doğruluğu:   0.886650025844574 
Test Doğruluğu:     0.8589000105857849 
Eğitim Kaybı:       0.31096816062927246 
Test Kaybı:         0.3975276052951813
'''
#overfitting ihtimalini azaltabilmek için bir drop out daha ekledim. eğitim doğruluğu ve test doğruluğu birbirine yaklaştı: 
''' 
Eğitim Doğruluğu:   0.8673833608627319
Test Doğruluğu:     0.8547999858856201
Eğitim Kaybı:       0.37093308568000793
Test Kaybı:         0.4052676260471344
'''
#8 epochtan sonra eğitim ve test verisi arasında fark artıyordu bu nedenle epoch değerini 8'e düşürdüm:
'''
Eğitim Doğruluğu:   0.8628666400909424
Test Doğruluğu:     0.8468000292778015
Eğitim Kaybı:       0.37758514285087585
Test Kaybı:         0.41748255491256714
'''
