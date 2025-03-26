import streamlit as st

st.title("Pemrosesan Beralur Big Data")
st.header("Flowchart")
#tampilkan gambar flowchart

st.image("arsitektur_bigdata.png", caption="Flow Big Data")

st.markdown('''
Proses awal big data dimulai pada sumber dari data itu sendiri. Sumber data dapat dberasal dari komunikasi antar manusia, komunikasi antar manusia dan mesin, komunikasi antar mesin, dan transaksi bisnis. Big data yang masuk juga bukan data yang kecil. Ukuran big data dapat lebih dari Terabyte. Sumber data pada big data biasanya disebut dengan ingest data yang berarti data yang 'ditelan' dan data input yang masuk sangatlah cepat. Ekosistem big data ada 3 tahap utama, yaitu data ingest, processing dan organizing data. Pada big data, processing data dibagi menjadi 2, yaitu stream processing dan batch processing. Stream processing adalah tahapan proses yang melakukan proses dari awal hingga akhir dengan sekaligus atau satu kali melakukan. Kekurangan dari stream processing adalah membutuhkan waktu yang lebih lama karena memproses big data dari awal hingga akhir. Batch processing adalah proses dengan data yang sudah dipartisi atau dipisah kedalam ukuran yang lebih kecil. Proses batch processing dapat melakukan processing dengan cepat, tetapi membutuhkan hardware yang lumayan kuat karena menjalankan banyak komputer kecil secara bersamaan. Hardware yang digunakan untuk memproses big data memerlukan tenaga yang besar.

Kelebihan dari batch processing adalah ketika data sudah diorganize, data dapat kembali lagi ke batch processing. Sementara stream processing hanya dapat satu kali, ketika data yang di-organize (yang sudah jadi) masih kurang, maka perlu me-run coding ulang dari awal. Ketitka data sudha di-organize, data dapat dilakukan analisis sesuai yang diinginkan, seperti melakukan data mining, visualisasi data, membuat dashboard, membuat laporan dan yang lain-lainnya.


Pada analisis, jenis processing yang digunakan adalah stream processing menggunakan spark. Spark adalah suatu mesin yang terintegrasi, cepat, in-memory (menggunakan memori) dan penggunaan general untuk preprocessing data yang besar. Spark dilengkapi dengan integrasi dengan python menggunakan `pyspark`.

''')

st.header("Data Event")

st.markdown("Data yang digunakan adalah data event atau kejadian dalam suatu e-commerce. Event yang dimaksud adalah kejadian dalam keranjang belanjaan belanja online, seperti pembelian, cancel, dan penambahan barang. Data event terdiri dari 13 variabel. Variabel tersebut adalah id, user_id, sequence_number, session_id, created_at, ip_address, city, state, postal_code, browser, traffic_source, uri dan event type. Berikut untuk kode untuk menyambungkan spark dengan `pyspark` ")

code_spark='''
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) # Property used to format output tables better
spark
'''
st.code(code_spark)

st.markdown('''
Setelah menyambungkan spark pada python, mari mengimport data event dengan coding di bawah ini.
''')

code_read = '''
df = spark.read.csv('events.csv', header=True, sep=",")
df.show(5) # menunjukkan 5 baris pertama
'''
st.code(code_read)

st.image("Data_view.png",
        caption="Tampilan 5 Data Pertama Event Suatu Ecommerce")


st.code("df.count()")
st.image("Data_count.png",
        caption="Banyaknya Data yang terdapat pada Data Event dari suatu E-commerce")
st.markdown('''
Data event dapat dicari jumlah baris dengan menggunakan `.count()`. Data event terdiri dari 2.434.151 baris. Data dimasukkan ke dalam spark dengan `pyspark`.
''')

st.markdown('''
Adapun arti dari setiap variabel:
- id : nomor urut id
- user_id : nomor unik user
- sequence_number : urutan keranjang
- created_at : tanggal event dilakukan
- ip_address : ip address user
- city : kota user berada
- state : provinsi user berada
- postal_kode : kode pos
- browser : user menggunakan browser apa untuk melakukan event
- traffic_source : asal traffic 
- uri : kode 
- event_type : tipe event

''')


st.subheader("Data Exploration with Spark")

st.markdown('''
Pada spark, kita bisa melakukan eksplor data. Untuk memahami data, kita akan melihat summary dengan `.summary()`, skema data dengan `.schema()`, dan kekosongan data.
''')

st.image("Data_summary1.png", caption ='Summary Data Event')

st.image("Data_schema.png", caption = 'Schema Data Event')

st.markdown(''' 
Pada summary data event, padat disimpulkan bahwa semua variabael merupakan data string atau karakter yang bisa diisi dengan NULL atau kosong. Tidak ada data yang numerik pada data, meskipun pada data summary, variabel id, userid, sequence number, dan postal code dapat dihitung. Selanjutnya, untuk melihat apakah terdapat data NULL, kosong atau NAN, dapat menggunakan fungsi sql pada pyspark. fungsi sql yang digunakan adalah `isnan`. Kita dapat memanggil fungsi sql agar bisa digunakan pada pyspark dengan memanggil `pyspark.sql.functions`. Berikut untuk pengecekan NULL pada data event.
''')
    
st.image("Data_null.na.png", caption= 'Pengecekan NULL atau NA pada setiap variabel')

st.markdown('''
Pada data event, hanya user_id saja yang memiliki nilai NULL, selain dari itu tidak ada. Untuk memastikan apakah NULLnya banyak, kita akan coba membuat tabel dengan variabel user id dan event type.
''')

st.image("Data_count_eventtype.png", caption = 'Tabel count dengan user id dan event type')

st.markdown('''
Pada data user id dan event, dapat dilihat banhwa banyak user idnya Null atau kosong. Hal ini disebabkan karena user id hanya muncul satu kali dan di bawahnya kosong, atau user hanya melihat-lihat saja tidak belanja. Dapat dilihat juga bahwa ketika user id NULL, tidak terdapat event purchase. Maka, dapat dikatakan nilai NULL tidak memengaruhi jumlah purchasing atau pembelian barang user.
''')

# unique variabels
st.markdown('''


### Unique Variabel
Pada setiap variabel, kita akan melihat dan mengecek apa isi dari setiap variabel beserta keunikannya. Pada pyspark, untuk melihat variabel unik, kita dapat menggunakan `distinct()`. Untuk mencari banyaknya suatu indikator dapat menggunakan `count()`.
''')
st.image("unique_browser.png", caption = "Unique Variabel Browser")

st.markdown('''
User mengakses e-commerce melalui browser. Browser memiliki 5 indikator, yaitu firefox, safari, chrome, IE dan browser lain-lainnya.  
''')

st.image("unique_eventtype.png", caption = "Unique Variabel Event Type")


st.markdown('''
Pada variabel event type e-commerce. Event type memiliki 6 indikator, yaitu cancel, purchase (belanja), cart (masuk ke dalam keranjang), department, home, dan product.
''')

st.image("unique_trafficsource.png", caption = "Unique Variabel Traffic Source")
st.markdown('''
Sumber traffic source dapa data event berasal dari Organic (pendekatan langsung), youtube, email, adwords (iklan) dan facebook. 
''')


st.image("unique_cityandstate.png", caption = "Banyaknya Kota dan Provinsi yang Unique")

st.markdown('''
Pada data event, user-user berasal dari 8767 kota yang berbeda, dan diantara kota tersebut terdapat 231 provinsi di seluruh dunia.
''')


st.image("unique_userid.png", caption = "Banyaknya User Id yang Unique")

st.markdown('''
Total user yang terdapat pada suatu e-commerce adalah 79.873 pengguna. 
''')


st.markdown(''' 
### Count and Histogram

Berikut tabel dan bar-plot untuk menjelaskan di setiap variabel. Pada subbab ini, akan menggunakan code `.groupBy()` untuk mencari apa yang akan dicari, code `.count()` untuk menghitung banyaknya, dan code `orderBy()` untuk mengurutkan mulai dari yang tertinggi atau terendah.

#### Browser
''')

st.image("Data_countbrowser.png")

st.image("barplot_browser.png")

st.markdown(''' 
Sumber browser pengguna menggunakan paling banyak adalah chrome dengan 1.211.439 baris. Perbedaan chrome dengan yang lainnya sangat pesat. Browser hrome lebih banyak diabndingkan safari dan firefox digabung. User kebanyakan menggunakan chrome untuk mengakses situs e-commerce. Adapaun browser lain yang digunakan berdasarkan urutan terbanyaknya adalah safari, firefox, browser lain, dan IE.

#### State
''')


st.image("Data_countstate.png")

st.image("barplot_state.png")

st.markdown('''  
Provinsi yang menyumbang kegiatan paling banyak pada data event e-commerce adalah provinsi Guangdong dengan 128.877 kegiatan e-commerce. Provinsi lainnya yang termasuk ke dalam 10 provinsi terbesar menyumbang kegiatan pada data adalah Guangdong, England, California, Texas, Shanghai, Sao Paolo, Zhejiang, Beijing, Hebei dan Jiangsu.

#### Traffic Source
''')

st.image("Data_counttraffic_source.png")

st.image("barplot_traffic.png")

st.markdown(''' 
Sumber traffic user sebagian besar berasal dari email dengan 1.091.794 kegiatan. Dapat disimpulkan bahwa user ke laman e-commerce biasanya berasal dari email. Adapun sumber traffic lainnya sesuai urutan banyaknya user ke lama e-commmerce, yaitu adwords (iklan), facebook, youtube dan organic.

#### Event 
''')

st.image("Data_countevent.png")

st.image("barplot_typeevent.png")

st.markdown(''' 
Sumber Traffic yang menyumbang kegiatan paling banyak pada data event e-commerce adalah traffic product dengan 843.683. Sumber traffic yang paling besar selanjutnya ada department, cart, purchase, cancel dan home. Pada sumber traffic ini, dapat dikatakan bahwa terdapat 181.293 kegiatan purchasing dan terdapat 124.446 order yang dicancel atau dihapus.
''')



st.subheader("Feature Engineering")

st.markdown('''
Pada subbab ini, data akan ditransformasi untuk mencari:
- banyaknya user yang melakukan purchasing
- mencari provinsi yang melakukan purchasing terbanyak pada e-commmerce
- mencari total user dan total purchase paling banyak dan paling sedikit berdasarkan provinsi
- mencari purchasing berdasarkan sumber traffic

untuk mencari semua poin-poin tersebut, kita akan menggunakan pyspark sql functions, seperti `groupBy`, `count` dan `sum`. Untuk mendapatkan total user dan total purchasing terbanyak berdasarkan provinsi, akan menambahkan menggabungkan data menggunakan fungsi `join`, fungsi join yang digunakan adalah `inner`, yaitu inner join. 

Alasan untuk memlilih hal-hal tersebut karena pada data event itu sendiri tidak memiliki variabel numerik. Variabel numerik yang ada bukan merupakan angka yang dapat diekstrak seperti ip address dan userid. Sisa dari variabel tersebut adalah variabel kategorik. Pada data ini, semua variabel menjelaskan tentang identitas kegiatan (event) yang sesuai pada user id. Maka dari itu, untuk variabel yang akan diekstrak adalah variabel identitas pengguna e-commerce agar dapat mengetahui karakteristik pengguna e-commerce, dengan menggunakan data tersebut e-commerce dapat melihat rencana kedepannya untuk melakukan campaign yang tepat dan penjualan yang meningkat.
''')

st.subheader("Spark Analysis")

st.markdown('''
Pada analisis spark, akan menggunakan dataframe pyspark. 
''')

st.image("a_user.totalpurchase.png", caption='Banyaknya User yang melakukan purchasing pada semua data')


st.markdown('''
Pada semua data, nilai maksimum seseorang yang sama melakukan purchasing adalah 13. Terdapat 3 orang yang melakukan pembelian terbanyak sebanyak 13 kali order. Kebanyakan setiap user melakukan purchase sebanyak 2 kali, banyaknya user yang melakukan purchasing 2 kali sebanyak 19.689 user. Jika dilihat pada tabelnya, ketika semakin banyak jumlah purchasing, semakin menurun jumlah orang yang melakukan order lagi. E-commerce harus dapat memberikan pelayanan atau fitur yang lebih baik lagi agar user dapat berbelanja terus. Penurunan banyaknya ornag yang melakukan purchase berulang turun semakin sedikit hingga di bawah 1000, dimulai pada purchase ke 7.  


Selanjutnya, untuk mencari provinsi dengan purchase terbanyak, dapat menggunakan `pyspark.sql.function` seperti, `sum`, dan juga akan membutuhkan fungsi `groupBy`. Adapun coding untuk mendapatkannya pada coding di bawah ini.  
''')



code_user_state ='''
from pyspark.sql.functions import sum
import pyspark.sql.functions as F

state= df.groupBy("user_id","state","event_type").count().filter(col('event_type')=='purchase')\
.groupBy("state").agg(sum("count").alias("total_purchase")).withColumnRenamed('state', 'stated')

state_user = df.groupBy("state","user_id","event_type").count().filter(col('event_type')=='purchase')\
.groupBy("state").agg(F.countDistinct("user_id").alias("total_user"))
'''

st.code(code_user_state)

st.image("a_state.totalpurchase.png", caption = "20 Provinsi dengan Purchasing terbanyak")

st.markdown('''
Provinsi dengan purchasing terbanyak terdapat pada provinsi Quandong. Dengan menggunakan tabel `state`, dapat melakukan join tabel dengan tabel `state_user` untuk mendapatkan data purchasing dengan provinsi dan jumlah usernya. Berikut tabel purchasing yang paling banyak dan sedikit berdasarkan total purchasing dan total user.
''')

st.image("a_low_statepurchase.png", caption = "10 Provinsi dengan Purchasing Paling Sedikit dengan Total User")

st.markdown('''
Provinsi yang memiliki purchasing paling sedikit ada pada provinsi Shimane dengan total purchasing 1 kali dan total user sebanyak 1. 10 Urutan purchasing berdasarkan total purchasing dan total user dari yang paling sedikit adalah Shimane, Nara, Iwate, Mie, Nagasaki, Shiga, Tochigi, Yamanashi, Akita dan Kochi.
''')

st.image("a_high_statepurchase.png", caption = "10 Provinsi dengan Purchasing Paling Banyak dengan Total User")

st.markdown('''
Provinsi yang memiliki purchasing paling banyak ada pada provinsi Guangdong dengan total purchasing 9666 kali dan total user sebanyak 4229. 10 Urutan purchasing berdasarkan total purchasing dan total user dari yang paling banyak adalah Guangdong, England, California, Texas, Shanghai, Sao Paolo, Zhejiang, Beijing, Hebei dan Jiangsu. Dari hal ini, dapat disimpulkan bahwa semakin banyak user biasanya jumlah purchasing akan meningkat. 


Selanjutnya, pada tabel di bawah ini akan membahas tentang purchasing berdasarkan traffic source.
''')

st.image("a_traffic.totalpurchase.png")

st.markdown('''
Purchasing terbanyak berasal dari sumber traffic adalah email dengan purchasing sebanyak 81.707. Urutan purchasing yang terbanyak adalah email, adwords, facebook, youtube dan organic. Dapat disimpulkan bahwa untuk campaign ke depannya e-commerce dapat melakukan campaign ke email dan adwords. jumlah purchasing email dan adwords sendiri sudah melebihi sebagian besar dari seluruh total purchasing. 
''')


st.subheader("Evaluation and Discussion")
st.markdown('''
Pada data event, banyak yang dapat disimpulkan untuk meningkatkan purchasing. Peningkatan purchasing dapat dilakukan dengan melakukan campaign pada 10 provinsi terbanyak dan ke 10 provinsi dengan user terbanyak. Selanjutnya e-commerce diharapkan dapat meningkatkan customer yang melakukan purchasing ulang. Dengan total user purchase, masih banyak customer yang tidak purchasing ulang >5 kali. Tim strategi dan marketig harus dapat meningkatkan customer melakukan purchasing. Hal ini dapat terjadi bisa karena adanya masalah yang masih belum tau alasannya. Sayangnya data event ini merupakan data tentang karakteristik kejadian dalam e-commerce saja, bukan data secara keseluruhan. Sehingga, data yang didapatkan hanya sebatas kejadian pada event, bukan membahas tentang barang yang dijual, dijual kemana, dan data diri user yang melakukan purchasing. 


Pada data event yang terdapat kurang lebih 2 juta record, data dimasukkan ke dalam spark dengan bantuan pyspark pada python. Adapun kelebihan dan kekurangan menggunakan spark. Dengan data yang sangat besar, spark dapat memasukkan data yang banyak dan proses data berlangsung dengan sangat cepat.  Spark dapat memproses dengan in-memory processing, sehingga lebih efisien dibandingkan frame work lainnya. Spark dapat diintegrasikan dengan yang lain, seperti pada analisis ini menggunakann python (pyspark). Spark juga dapat diintegrasikan dengan R, SQL, java dan scala.  Kekurangan dari spark untuk pemula adalah membutuhkan pembelajaran dan pembiasaan ulang mengenai bahasa atau coding yang digunakan pada spark. Karena spark menggunakan in-memory processing, maka akan membutuhkan tenaga yang kuat yang mana hardware yang diperlukan harus kuat. Spark juga menghasilkan tabel yang hanya bisa dilakukan pada spark, pada analisis disini, tabel yang dihasilkan spark perlu diconvert menjadi dataframe dengan pandas. Dengan segala kelebihan dan kekurangan spark, spark dapat digunakan untuk menganalisis data yang banyak karena spark dapat membantu menganalisis big data.



#### Thankyou
Alicia Arifin
Statistika / Matana University
''')





