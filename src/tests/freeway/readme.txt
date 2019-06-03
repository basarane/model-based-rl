20190517 (initial defaults: Nadam, lr=0.0001
  output_e01 same count of each sprite in all batches. [(64,3),(64,3),(32,5),(32,5),(32,5),(64,3),(64,3),(1,3)]
  output_e02 same run previous
  output_e03 same run previous
  output_e04 5'lik kernel'lar 3 yapildi [(64,3),(64,3),(32,3),(32,3),(32,3),(64,3),(64,3),(1,3)]
  output_e05 Nadam'dan adam'a gecildi - gecilmemis'de olabilir. model yerine model_encoder'in optimizer'ini degistirmisim
  output_e06 Nadam'a geri donuldu, ReLU'dan ELU'ya gecildi
  output_e07 ELU'dan tanh'a gecildi, son layer RELU birakildi
  output_e08 RELU'ya geri donuldu, kernel size 4'e cikartildi
  output_e09 tum layerlarda filter sayisi 64 yapildi
* output_e10 ilk 3 layer 64 birakildi, son 4 layer 32 yapildi
  output_e11 same run previous: [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)] + RELU + Nadam
  output_e12 [(128,4),(64,4),(64,4),(32,4),(32,4),(16,4),(16,4),(1,4)]
  output_e13 [(128,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  **** e18'ye kadar yanlislikla model_encoder'in optimizer'ini degistirmisim
  output_e14 Nadam epsilon yukseltildi (e=0.1) [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e15 Nadam epsilon degisti (e=0.001)
* output_e16 same run previous (bu e10 ile aslinda ayni, epsilon degismedi, default)
  output_e17 Nadam epsilon degisti (e=0.0001)
  output_e18 Nadam epsilon degisti (e=0.01)
  output_e19 Nadam epsilon (e=0.1), [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e20 Nadam epsilon (e=0.001), [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e21 Nadam epsilon (e=0.0001), [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
* output_e22 Nadam epsilon (e=0.00001), [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]  ***** converge ediyor ama baya yavas
  output_e23 Nadam epsilon (e=0.000001), [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
* output_e24 Nadam epsilon (e=0.000001), [(128,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e25 Nadam epsilon (e=0.0000001 keras default), [(128,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e26 same run with output_e24: Nadam epsilon (e=0.000001), [(128,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e27 Nadam epsilon (e=0.000001), [(256,4),(128,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e28 Nadam epsilon (e=0.0000001 keras default), [(256,4),(128,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e29 Nadam lr=0.002 others=defaults [(256,4),(128,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e30 Nadam lr=0.00001 others=defaults [(256,4),(128,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e31 Nadam lr=0.00003 others=defaults [(256,4),(128,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e32 Nadam lr=0.00003 others=defaults [(256,3),(128,3),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e33 Nadam lr=0.0001 others=defaults [(256,3),(128,3),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e34 Nadam lr=0.0001 others=defaults [(256,4),(128,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e35 e10 ve e16 tekrari Nadam lr=0.0001 others=defaults [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  output_e36 ayni
  output_e37 ayn
* output_e38 ayni
  output_e39 ayni (batchnorm kodu eklenip disable edildi)
  output_e40 ayni 
  output_e41 ayni 
  output_e42 ayni 
  output_e43 ayni 
  output_e44 ayni 

20190519 - model ogrenmesi output_e38 kullanildi
defaults: optimizer = Nadam(lr=0.0001), [(cc,15)], elu, RandomNormal(mean=0.0, stddev=0.001, seed=None), mse
  output_m1 defaults
  output_m2 loss=100*mse
  output_m3 loss=mse, Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
  output_m4 same previous, 10k step instead 1k
  output_m5 same previous, real/estimated for all-steps saved, network saved
  output_m6 [(64,4),(64,4),(cc,11)]
  output_m7 [(cc,15)] glorot_normal
  output_m8 [(64,4,'relu'),(64,4,'relu'),(cc,15,'elu')]
  output_m9 manual weight test
  output_m10 manual weight test
  output_m11 manual weight test loss sadece arabalardan geliyor
  output_m12 manual weight test loss sadece 3 ve 4. arabadan geliyor (deterministic olanlar). loss'un 0 olmasini bekliyorum
  output_m13 oncekinin aynisi. ama 3 ve 4 degil 4'den 7'ye deterministicmis
  output_m14 [(cc,57,'elu')], ACTION_REPEAT=4, normal loss, learning weights
  output_m15 [(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(cc,4,'relu')]
  output_m16 [(128,4,'elu'),(64,4,'elu'),(64,4,'elu'),(64,4,'elu'),(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(32,4,'elu'),(cc,4,'relu')]
  output_m17 [(256,4,'elu'),(128,4,'elu'),(64,4,'elu'),(64,4,'elu'),(64,4,'elu'),(64,4,'elu'),(64,4,'elu'),(64,4,'elu'),(64,4,'elu'),(cc,4,'relu')]
  output_m18 crazy scientist [(Conv2D,64,4,2,'elu'),(Conv2D,64,4,2,'elu'),(Conv2D,64,4,2,'elu'),(Conv2D,64,4,2,'elu'),(Conv2D,32,4,2,'elu'),(Reshape,1,1,1120,'elu'),(Conv2DTranspose,128,(4,3),2,'elu'),(Conv2DTranspose,128,4,2,'elu'),(Conv2DTranspose,128,4,2,'elu'),(Conv2DTranspose,128,(7,4),2,'elu'),(Conv2DTranspose,128,(6,4),2,'elu'),(Conv2DTranspose,128,(4,3),2,'elu'),(Conv2DTranspose,cc,(5,4),1,'relu')]
  output_m19 full manual ACTION_REPEAT=4 test
  
20190521 - encoder artik alpha channelli ve overlapping objelerle egitilecek
  test_run_02 Adam (amsgrad=True), [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  test_run_03 Nadam
  test_run_04 same
  test_run_05 ??
  test_run_06 ??
  test_run_07 [(128,4),(128,4),(64,4),(64,4),(64,4),(64,4),(64,4),(1,4)]
  test_run_08 [(128,4),(64,4),(64,4),(32,4),(32,4),(16,4),(16,4),(1,4)]
  test_run_09 [(32,4),(32,4),(32,4),(64,4),(64,4),(64,4),(128,4),(1,4)]
  test_run_10 [(128,4),(128,4),(128,4),(128,4),(128,4),(128,4),(128,4),(1,4)]
  test_run_11 [(128,4),(128,4),(128,4),(128,4),(128,4),(1,4)]
  test_run_12 [(32,4),(32,4),(32,4),(32,4),(32,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  test_run_13 ELU
  test_run_14 LeakyReLU
  test_run_15 PReLU
  test_run_16 [(64,4),(64,4),(64,4),(64,4),(64,4),(32,4),(32,4),(32,3),(32,3),(1,3)] - car4'un lastikleri eklendi
  test_run_17 [(128,4),(64,4),(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  test_run_18 Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
  test_run_19 [(128,7),(64,6),(64,5),(64,4),(64,3),(32,3),(32,2),(32,2),(32,2),(1,2)] Nadam(lr=0.0001, )  
  test_run_20 [(128,5),(64,5),(64,4),(64,4),(64,4),(32,4),(32,4),(32,3),(32,3),(1,3)] 
  test_run_21 [(128,5),(64,5),(64,5),(64,5),(64,4),(1,4)]
  test_run_22 TrainNetwork.ENCODER: model yerine model_encoder direk train edildi
  test_run_23 amsgrad
  test_run_24 elu
  test_run_25 [(64,3),(64,3),(64,3),(64,3),(64,3),(64,3),(64,3),(64,3),(64,3),(1,3)] + ReLU + nadam + TrainNetwork.CHANNELED_AUTO_ENCODER
  test_run_26 same
  test_run_27 [(64,4),(64,4),(64,4),(32,4),(32,4),(32,4),(32,4),(1,4)]
  test_run_28 enable weight sharing [(128,4, True),(64,4, True),(64,4, False),(32,4, False),(32,4, False),(32,4, False),(32,4, False),(1,4, False)] 
* test_run_29 enable single last layers [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
* test_run_30 same
* test_run_31 [(128,4, True, False),(64,4, True, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_32 [(128,4, False, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_33 [(128,4, False, True),(64,4, False, True),(32,4, False, True),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
* test_run_34 [(128,4, True, False),(64,4, True, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_35 [(256,4, True, False),(128,4, True, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_36 [(64,4, True, False),(32,4, True, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_37 [(64,4, True, False),(64,4, True, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_38 [(128,4, True, False),(64,4, True, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_39 [(128,4, True, False),(64,4, True, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
* test_run_40 [(128,4, True, False),(64,4, True, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(16,4, False, False), (32,3,False,True), (cc,4, False, True)]
* test_run_41 [(128,4, True, False),(64,4, True, False),(32,4, False, False),(16,4, False, False),(16,4, False, False),(16,4, False, False),(8,3, False, False),(8,3, False, False),(8,3, False, False), (32,3,False,True), (cc,4, False, True)]
* test_run_42 [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(16,3, False, False),(16,3, False, False),(16,3, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_43 [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(16,3, False, False),(16,3, False, False),(16,3, False, False), (32,3,False,True), (32,3,False,True), (cc,4, False, True)]
  test_run_44 [(128,4, True, False),(64,4, True, False),(64,4, True, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(16,3, False, False),(16,3, False, False),(32,3,False,True), (cc,4, False, True)]
  test_run_45 [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(8,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_46 [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(4,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_47 [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(2,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_48 [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(16,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)]
  test_run_49 [(128,4, True, False),(64,4, True, False),(64,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(32,4, False, False),(1,4, False, False), (32,3,False,True), (cc,4, False, True)] same with 29 
* test_run_50 [(128,3, True, False),(64,3, True, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (32,3,False,True), (cc,3, False, True)]
  test_run_51 [(128,3, True, False),(64,3, True, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (32,3,False,True), (cc,3, False, True)]
  test_run_52 [(128,3, True, False),(64,3, True, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (32,3,False,True), (cc,3, False, True)]
  test_run_53 network: same with 50. PReLU
  test_run_54 network: same with 50. LeakyReLU
  test_run_55 network: same with 50. ReLU
  test_run_56 ayni anda iki sprite train ediliyor: car + chicken. Overlap var. 
  test_run_57 sample'larin sadece 30%'inde 2 sprite (overlap) var. 
  test_run_58 [(256,3, True, False),(128,3, True, False),(64,3, False, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (32,3,False,True), (cc,3, False, True)]
  test_run_59 [(256,3, True, False),(128,3, True, False),(64,3, False, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (64,3,False,True), (64,3,False,True), (cc,3, False, True)]
  test_run_60 [(512,3, True, False),(256,3, True, False),(128,3, False, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (64,3,False,True), (64,3,False,True), (cc,3, False, True)]
  test_run_61 yeni bilgisayarda 60'in tekrari
  test_run_62 58'in tekrari [(256,3, True, False),(128,3, True, False),(64,3, False, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (32,3,False,True), (cc,3, False, True)]
  test_run_63 58'in tekrari
  test_run_64 58'in tekrari, uretilen train sample'lari kontrol etmek icin calistirildi
  test_run_65 58'in tekrari. training samplelarindaki problem duzeltildi ( == 0 <=0.0001 yapildi). 
  test_run_66 58'in tekrari. tavuk ihtimali %50'ye cikartildi. x range'i 5,-6'dan 0,-1'e cikartildi. sadece arabalara tavuk eklendi (galiba onceden tavuklara da ekleniyormus)
  test_run_67 58'in tekrari. get_large_decoder'daki layer'lari birlestirme add'den custom layer merge_tensors'a degistirildi
  

* test_run_68 58'in tekrari. uzun run: 20000
  test_run_m69 Model ogrenmeye gecildi. [(Conv2D,cc,57,1,'elu')] output_m14'nin yeni encoder ile tekrari. en basit model
  test_run_m70 [(Conv2D,cc,15,1,'elu')] ACTION_REPEAT=1
  test_run_m71 [(Conv2D,cc,13,1,'elu')]
  test_run_m72 [(Conv2D,cc,9,1,'elu')]
  test_run_m73 [(Conv2D,cc,5,1,'elu')]
  test_run_m74 [(Conv2D,cc,7,1,'elu')]
  test_run_m75 [(Conv2D,cc,8,1,'elu')]
  test_run_m76 [(Conv2D,cc,9,1,'relu')]
  test_run_m77 same with previous
  test_run_m78 same with previous, initial weights are manual
  test_run_m79 same with previous, initial weights are manual, Adam(amsgrad=False)
  test_run_m80 same with previous, initial weights are manual, Nadam
  test_run_81 encoder learn. same with test_run_68. merge layer is changed to alpha compositing (https://en.wikipedia.org/wiki/Alpha_compositing)
  test_run_82 alpha=0 oldugu zamanki problemi cozmeye calistim 
  test_run_m83 test_run_82'deki encoder ile test_run_m80'in tekrari
  test_run_m84 test_run_82'deki encoder ile test_run_m79'in tekrari (Nadam => Adam)
  test_run_m85 loss hesaplarken x,y'den alt ve ustten 10px kirpildi, tavuk cikartildi
  test_run_m86 oncekinin aynisi, initial weight'ler random (onceki manuel'di)
  test_run_m87 tavuk geri katildi (onun convolution'i yuzunden goruntu bozuluyordu)
  test_run_m88 RandomUniform(minval=0.05, maxval=0.10, seed=None)
  test_run_m89 RandomUniform(minval=0.005, maxval=0.01, seed=None)
  test_run_m90 same with previous 20000 step
  test_run_m91 ACTION_REPEAT=4 [(Conv2D,cc,33,1,'relu')]
  test_run_m92 action secim olasiliklari [0.2,0.6,0.2]
  test_run_m93 filtreler (1,33) ve (33,1) olarak degistirildi

20190528
  test_run_m94 full custom network, her bir araba icin ayri convolution. 3 tavuk arabalarin merged state'leri ile birlesip action ile multiplied oluyor. sonra da convolution'lar var.
  test_run_m95 sonraki network'lere comet.ml'den bak. ozet: 32x(33,33) 32x(33,33) 3(33,33) randomuniform
  test_run_m96 32x(33,33) 32x(1,1) 3(1,1)
  test_run_m97 3x(33,33) tek convolution
  test_run_m98 cars_mixed kullanilmiyor, sadece tavuk, yani carpisma modeli yok
  test_run_m99 cars_mixed_next ve cars_mixed_current 1x33x33'luk conv'dan gecirildi
  test_run_m100 cars_mixed_next cikartildi, ogrenmeyi zorlastiriyordu. belki trainable=false denilerek tekrar denenebilir
  test_run_m101 ACTION_REPEAT=1 33'luk filtreler 9 yapildi
  test_run_m102 test_run_m100'un aynisi (kontrol et). ACTION_REPEAT=4, 9'luk filtreler tekrar 33. araba conv'larinin inputu pad'lendi, ust 5 arabanin soldan cikanlar saga eklendi (16'lik genisledildi)
  test_run_m103 8 pixel kayiyordu, duzeltmeye calistim
  test_run_m104 seed=100, manual weights, ACTION_REPEAT=4, original network: [[(Conv2D,cc,(33,33),1,'relu')]]
  test_run_m105 aynisi 10000 looplu. trainable=false ise yariyor mu diye bakiyorum
  test_run_m106 200 step learning yok, her step kaydoluyor, encoded'lar yazilmiyor. tavuk eklendi. full manuel'e geciyorum
  test_run_m107 learning tekrar aktif edildi, trainable compile'dan onceye alindi, 105'in tekrari gibi
  test_run_m108 aynisi, trainable=False kaldirildi
  test_run_m109 aynisi ama bazi arabalar kaybolmustu, 104'de duzgundu. manuel weight'lerden sadece 13+10 ve 13+12'birakildi
  test_run_m110 yine ayni arabalar yok, sadece arabalar birakildi
  test_run_m111 tam nedenini anlamadim, kodu temizledim, split ve birlestirme ile ilgili kismi kaldirdim.
  test_run_m112 encoded image'larini tekrar kaydediyorum
  test_run_m113 encoded image kaydetmesi sirasinda <0 => 0 ve >1 => 1 yapiliyormus, sadece onu iptal ettim.
  test_run_m114 kafam karisti tekrar save_image encode'lari iptal ettim (sonuc: encoded save_image'i kaldirinca gercekten arabalar kayboluyor)
  test_run_m115 sadece save_image(next_est...'i biraktim
  test_run_m116 save_image encoded'in basina np.copy ekledim (sonuc: tahmin ettigim gibi arabalar kayboldu tekrar)
  test_run_m117 prediction image'ini kaydederken <0 ve >1 kontrolu yokmus, ekledim. 
  test_run_m118 seed test 1
  test_run_m119 seed test 2 (sonuc: iki run'da ayni sonuclar aliniyor)
  test_run_m120 1_1, 1_3, 1_6, 2_1'de tavukta bazi pixeller bos geliyor >1 yerine carpmadan sonra >255 ekledim 
  test_run_m121 decoder networkunun kodu model'in icine alindi, weight'i load edilenden set edildi. division by zero'daki +1 fixi yerine +0.001 yapildi (sonuc: tavuktaki bos pixel problemi cozuldu)
  test_run_m122 yukari cikan tavuk ekranin altina gelecek sekilde manuel kod eklendi
  test_run_m123 ustu keseyim derken solu kesmisim yanlislikla (h x w x c)
  test_run_m124 alta inenler de duzeltildi, alttaki tavugun pozisyonu duzeltildi
  test_run_m125 10k'lik egitim baslatildi
  test_run_m126 tekrar ACTION_REPEAT=1'e donuldu
  test_run_m127 son 3 arabada problem vardi, elle duzeltildi
  test_run_m128 bottom_crop 15'den 17'e cikartildi
  test_run_m129 env.action_space.np_random.seed eklendi
  test_run_m130 2000 step'e cikartildi
  test_run_m131 cross_timer eklendi, batch_size=1 yapildi. bu cross_timer'in bir sample'in bir onceki sample'a bagli oldugu icin yapildi
  test_run_m132 cross_timer'daki hatalar cozuldu, 2000'lik run alindi
  test_run_m133 cross_timer_count=7 yapildi (once 5 sonra 6 yapmistim)
  
20190602
  test_run_m134 carpisma checki eklendi: pipenv run python model_learn.py test_run_68\ 6000 test_run_m134
  test_run_m135 carpisma hep 0 dondu, next_obs yerine input kullanildi ve toplanmis alpha channeli resim olarak kaydedildi. sonuc: resimler bos geldi
  test_run_m136 channels networku direk yuklendi, set weights yapilmadi
  test_run_m137 sadece alpha channeli degil, tum channellar toplandi, switch'den gecirilmeden kaydedildi
  test_run_m138 tek araba ve tek tavuk direk slice edildi
  test_run_m139 networkun output'unu degil, kendi yarattigim degiskeni kaydediyormusum (output yanlislikla test_run_m138'in uzerine kaydedildi). sonuc: ok, sorun yokmus
  test_run_m140 tekrar alpha channelina gectim: sonuc: sadece 1. araba gozukuyor
  test_run_m141 rgba'nin tamamini topladim. sonuc: araba+tavuk gozuktu
  test_run_m142 sadece r'leri topladim
  test_run_m143 keras strided slice'i direk testeklemiyormus. tf.strided_slice ile degistirdim
  test_run_m144 batch dim'ini atlamisim. batch_size=1 gibi kodladim direk
  test_run_m145 check_carpisma strided_slice ile degistirildi
  test_run_m146 model tamamlandi, ilk test. bazi debug print'leri cikartildi
  test_run_m147 aynisi sadece carpisma_timer print edildi
  test_run_m148 sorunlar cozuldu, learning'in durumunu gormek icin ayni run tekrar yapiliyor. 10000 step'e cikartildi
  test_run_m149 training iptal edildi, network kaydediliyor
  test_run_m150 reward eklendi
  
 denenecekler: 
  - tavuk'un karsiya gecisinin ve kazanin count'unu ayri iki degiskende tut
  - action'lar arabayi etkilemiyor, onlari ayri tahmin et, birlestirilmis ve extend edilmis hallerini action+tavukla birlestir
  - search space'i kucultmek icin conv filtreleri 33x1 ve 1x33 seklinde degistir
  - problem arabanin carptigini anlama zorlugu, arabalarin layerlarini birlestir uzerine 9x9 conv koy, aynisini tavuga da yap, sonra bu iki map'i carp, ciktiyi topla, tek sayiya indir :)
  - alttaki iki problemi de inputu genisleterek ve output hesaplarken ustu alta, sol/ustu sag/uste, sag/alti sol/alta alarak cozebilir miyiz?
	- tavugun yukaridan asagiya gecis problemi
	- arabalarin ekrandan cikinca diger taraftan cikmasi (bunu yapmaya gerek var mi acaba?)
  + chicken car overlap'i dene
  + ELU+amsgrad
  + elu+amsgrad+encoder.compile
  + ilk layer'lari veya son layer'lari share et, son layerlarda birlesmis daha uzerine conv uygula
