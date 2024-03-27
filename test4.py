import pandas as pd
import spacy
from spacy import displacy
import re

nlp = spacy.load("en_core_web_sm")

traindata = [
    "16 April 1990 – The Patna rail disaster involved a shuttle train consumed by fire near Patna, killing 70.",
    "6 June 1990 – A fire on a train at Gollaguda, in Andhra Pradesh, killed 36 people.",
    "25 June 1990 – A freight train collided with a passenger train at Mangra, Bihar, leading to 60 deaths.",
    "10 October 1990 – A train fire near Cherlapalli in Andhra Pradesh killed 40 people.",
    "6 March 1991 – The Karnataka Express derailed in the rain near Makalidurga ghats, about 60 km (37 mi) from Bangalore, killing 30.",
    "12 December 1991 – The Kangra valley train suffered a crash near Jawali, Himachal Pradesh, killing 27.",
    "5 September 1992 – A collision near Raigarh in Madhya Pradesh killed 41.",
    "16 January 1993 – The Howrah Rajdhani Express collided with a goods train between Roorah and Ambiyapur flag stations.",
    "16 July 1993 – A train accident in the Darbhanga district of Bihar killed 60.",
    "21 September 1993 – A Kota–Bina passenger train collided with a freight train near Chhabra in Rajasthan, killing 71.",
    "3 May 1994 – The Narayanadri Express collided with a tractor in Nalgonda district of Andhra Pradesh, killing 35.",
    "26 October 1994 – The 8001 DN Mumbai–Howrah Mail's sleeper coach caught fire in the early morning, killing 27.",
    "14 May 1995 – The Madras–Kanyakumari Express collided head-on with a freight train near Salem, killing 52.",
    "1 June 1995 – The Jammu Tawi Express from Kolkata collided with a stationary coal-laden goods train, killing 45 and injuring 335.",
    "1 June 1995 – Four carriages of the Hirakud Express derailed and fell down an embankment at Barpali, Orissa, causing 15 deaths.",
    "20 August 1995 – The Firozabad rail disaster occurred when the Purushottam Express collided with the stationary Kalindi Express near Firozabad, killing 400.",
    "18 April 1996 – A Gorakhpur–Gonda passenger train collided with a stationary freight train at Domingarh station near Gorakhpur, Uttar Pradesh, killing 60.",
    "14 May 1996 – An Ernakulam–Kayamkulam train collided with a bus at an unmanned level-crossing near Alappuzha, Kerala, killing 35 people in a marriage party.",
    "25 May 1996 – An Allahabad-bound passenger train collided with a tractor-trolley at an unmanned level-crossing near Varanasi, killing 25.",
    "30 December 1996 – The Brahmaputra Mail train bombing occurred when the Brahmaputra Mail was blasted by a bomb between Kokrajhar and Fakiragram stations in lower Assam, killing 33.",
    "8 July 1997 – A passenger train was blasted by a bomb at Lehra Khanna railway station, Bhatinda district, Punjab, killing 33.",
    "28 July 1997 – The Karnataka Express and Himsagar Express collided on the outskirts of Delhi, killing 12.",
    "15 August 1997 - 42 Down Coromandel Express collided head-on with 41 Up Coromandel Express near Visakhapatnam in Andhra Pradesh killing 75.",
    "14 September 1997 – Five cars of the Ahmedabad–Howrah Express plunged into a river in Bilaspur district, Madhya Pradesh, killing 81.",
    "5 October 1997 - 42 Down Coromandel Express derailed at Rajahmundry after crossing Godavari (Havelock) Bridge. The bridge was decommissioned after the incident & New Arch Bridge was thrown open for rail traffic.",
    "6 December 1997 - Serial bomb blasts in Cheran Express at Erode, Pandian Express at Tiruchirappalli and Chennai-Alleppey Express at Thrissur kills 10 people and injures 70.",
    "6 Jan 1998 – A Bareilly–Varanasi passenger train collided with the stationary Kashi Vishwanath Express about 12 km (7.5 mi) from Hardoi, killing 70.",
    "4 April 1998 – The Fatuha train crash occurred when the Howrah–Danapur Express derailed near Fatuha Station on the Howrah–Delhi main line, killing 11 and injuring 50.",
    "24 April 1998 – 15 cars of a freight train collided with the Manmad–Kachiguda Express at Parli Vaijanath railway station in Maharashtra, killing 24 and injuring 32.",
    "13 August 1998 – The Chennai–Madurai Express train collided with a bus at an unmanned level-crossing on the new Karur-Salem bypass road on the outskirts of Karur town, killing 19 and injuring 27.",
    "24 September 1998 – A locomotive collided with a bus at an unmanned level-crossing near Bottalaapalem village in Andhra Pradesh, killing 20 (including 14 school children) and injuring 33.",
    "26 November 1998 – The Khanna rail disaster occurred when the Jammu Tawi–Sealdah Express collided with three derailed coaches of the Frontier Golden Temple Mail at Khanna, killing over 212 people.",
    "16 July 1999 – The 2616 UP Chennai–New Delhi Grand Trunk Express collided with derailed cars of a DN freight train on the Agra-Mathura section of the Central Railway, killing 17 and injuring over 200.",
    "2 August 1999 – The Gaisal train disaster occurred when the Brahmaputra Mail collided into the stationary Avadh Assam Express at Gaisal station in North Frontier Railway's Katihar division, killing at least 285 and injuring more than 300.",
    "15 August 1999 - Coromandel Express derailed at Dusi, just crossing Nagavalli River killing 50 passengers and injuring 500.",
    "6 December 1999 - Coromandel Express derailed at Jenapur, killing 20 passengers and injuring 100. Tracks were damaged due to Super Cyclone.",
    '2 December 2000 – The Howrah Amritsar Mail express collided with derailed coaches of goods train between Ambala Ludhiana. Over 45 people were killed and 150 were injured.',
    '22 June 2001 – The Kadalundi train derailment occurred when the Mangalore–Chennai Mail commuter train was crossing Bridge 924 over the Kadalundi River and four carriages derailed and fell into the river, killing 52 and injuring as many as 300.',
    '27 February 2002 – The Sabarmati Express was attacked by a mob at Godhra station in Gujarat and four coaches were set on fire, killing 58 and wounding 43.',
    '28 February 2002 - In retaliation for the 27 February Sabarmati Express case, the Ahmedabad - Delhi Ala Hazrat Express was attacked by a mob, injuring 100-200 passengers.',
    '13 May 2002 – The 2002 Jaunpur train crash occurred when sabotage derailed the Shramjeevi Express at Jaunpur, Uttar Pradesh, killing 12 and injuring 80.',
    '4 June 2002 – The Kasganj level crossing disaster occurred when a Kanpur–Kasganj Express collided with a passenger bus near the town of Kasganj, Uttar Pradesh, killing 30 and injuring 29.',
    '9 September 2002 – The Rafiganj train wreck occurred when the Howrah Rajdhani Express derailed on a bridge between Gaya and Dehri-on-Sone stations, with two coaches falling into a river, killing over 140. Terrorist sabotage was blamed.',
    '15 May 2003 – The Golden Temple Mail caught on fire in the early morning between Ludhiana and Ladhowal stations, killing 36 and injuring 15.',
    '22 June 2003 – The engine and first four coaches of Karwar–Mumbai Central Holiday Special derailed, killing 52 and injuring 26.',
    '2 July 2003 – The Golconda Express derailed at Warangal station, killing 21 and injuring 24.',
    '16 June 2004 – The Matsyagandha Express (Mangalore–Mumbai) derailed when it struck a boulder on the line, killing 14.',
    '14 December 2004 – The Jammu Tawi Express collided head-on with the DMU Jalandhar–Amritsar passenger train about 10 km (6.2 mi) from Mukerian town, Hoshiarpur, Punjab, killing 37 passengers and injuring 50. The accident was allegedly caused due to a fault in the telephone line, which prevented proper signal warnings.',
    '28 July 2005 – The 2005 Jaunpur train bombing occurred when a bomb destroyed a carriage of the Shramjeevi Express near Jaunpur, Uttar Pradesh, killing 13 and injuring 50.',
    '3 October 2005 – The Datia rail accident occurred when the Bundelkhand Express, travelling at excessive speed, overshot a sharp turn and derailed near Datia, killing 16 and injuring more than 100.',
    '25 October 2005 – The Island Express had several coaches derail near Kamasamudram on the Bangalore–Jolarpettai section.',
    '29 October 2005 – The Valigonda train wreck occurred when the Delta Fast Passenger train derailed where a small rail bridge had been swept away by a flash flood, near the town of Valigonda, Andhra Pradesh, killing at least 114 and injuring over 200.',
    '11 July 2006 – The 2006 Mumbai train bombings were a series of coordinated bomb attacks on commuter trains in Mumbai, killing at least 200 and injuring over 700.',
    '20 November 2006 – The 2006 West Bengal train explosion on a train near Belacoba station in West Bengal, suspected to be a terrorist bombing, resulted in 7 killed and 53 injured.',
    "1 December 2006 – The 150-year-old 'Ulta Pul' bridge in Bihar was being dismantled when a portion collapsed onto a passing Eastern Railways train, the Howrah–Jamalpur Super fast express (13071 up), killing 35 and injuring 17.",
    '18 February 2007 – The Delhi–Lahore Samjhauta Express was attacked by terrorists, causing 68 deaths.',
    '7 August 2007 – 32 passengers were injured when the Jodhpur–Howrah Express derailed near Juhi Bridge, Kanpur.',
    '1 August 2008 – The Gowthami Express caught fire while crossing Kesamudram station in Andhra Pradesh, killing 40. The fire was supposedly caused by an electrical short circuit.',
    '13 February 2009 – Several carriages of the Coromandel Express caught fire after leaving Jajpur Road station in Orissa.',
    '29 April 2009 – An Electric Multiple Unit of Southern Railways collided with empty oil tanker of a goods train at Vyasarpadi Jeeva station following a hijack, killing 4.',
    '20 October 2009 – The Mathura train collision occurred when the Goa Express collided with the rear carriage of the stationary Mewar Express outside Mathura, Uttar Pradesh, killing 21 and injuring several others. An investigation determined there was a faulty signal.',
    '2 January 2010 – The 2010 Uttar Pradesh train accidents were a series of three accidents that took place in Uttar Pradesh due to dense fog conditions.',
    'The Lichchavi Express collided with the stationary Magadh Express in a station near Etawah, about 270 km (170 mi) southwest of Lucknow.',
    'The Gorakhdham Express and Prayagraj Express collided near Panki station in Kanpur, about 97 km (60 mi) southwest of Lucknow, killing 10 and injuring 51.',
    'The Saryu Express hit a tractor trolley at an unmanned railway crossing in Pratapgarh. There were no injuries.',
    '3 January 2010 – Arunachal Pradesh Express derailed near Helem. No injuries were reported.',
    '16 January 2010 – Kalindi–Shram Shakti collision, near Tundla.',
    '17 January 2010 – Harihar–car collision at Barha railway crossing under Haidergarh police station area in Barabanki district.',
    "22 January 2010 – Goods train derailment near Azamgarh.",
    "25 May 2010 – A Rajdhani Express train derailed in Naugachia, Bihar, injuring 11. The train derailed as the driver applied emergency brakes after hearing a loud explosion.",
    "28 May 2010 – The Jnaneswari Express train derailment occurred when the Howrah–Lokmanya Tilak Terminus Jnaneswari Super Deluxe Express was derailed by an explosion then struck by an oncoming freight train between Khemashuli and Sardiha stations, killing at least 140 and injuring about 200. Maoists were suspected in the attack.",
    "4 June 2010 – The Coimbatore–Mettupalayam special train collided with a mini-bus at an unmanned level-crossing at Idigarai near Coimbatore, killing 5.",
    "18 June 2010 – The 8084 (Vasco-da Gama–Howrah) Amaravati Express derailed after colliding with a road-roller at an unmanned level crossing near Koppal, Karnataka, injuring 27.",
    "19 July 2010 – The Sainthia train collision occurred when the Uttar Banga Express collided with the Vananchal Express at Sainthia railway station killing 66.",
    "8 August 2010 – Four people, including two foreigners, killed when the Chennai-Alappuzha superfast express collided with a car in an unmanned railway gate near Mararikulam on Ernakulam – Alappuzha coastal line.",
    "17 August 2010 – A train accident[specify] occurred at Goryamau railway station of Barabanki district, killing 4.",
    "1 January 2011 – Akaltakth–trucks collision at Babura railway crossing in Jaunpur district.",
    "3 January 2011 – Goods train derailment at Dadri area of Ghaziabad district.",
    "7 July 2011 – A Mathura–Chhapra Express hit a bus at an unmanned level crossing in Thanagaon, Kanshiram Nagar district, Uttar Pradesh, killing 38 and injuring 30.",
    "10 July 2011 – The Fatehpur derailment occurred when the Kalka Mail derailed near Fatehpur, Uttar Pradesh, killing 70 people and injuring more than 300.",
    "10 July 2011 – The Guwahati–Puri Express derailed in Nalbari district, Assam, with the engine and four coaches rolling over in a rivulet, injuring more than 100. The derailment was determined to have been the result of a bomb attack by local militants.",
    "23 July 2011 – A second freight train had eight cars derail at almost the same place as the previous incident, less than 24 hours earlier.",
    "31 July 2011 – The Guwahati–Bangalore Kaziranga Express derailed and was hit by another train in Malda district, West Bengal, killing at least 3 and injuring 200.",
    "13 September 2011 – A Chennai suburban MEMU (Mainline Electric Multiple Unit) commuter train collided with a stationary Arakonam–Katpadi passenger train between Melpakkam and Chitheri stations in Vellore district, killing 10 and injuring many others.[121]",
    "22 November 2011 – A Howrah–Dehradun Doon Express caught fire in two coaches, killing 7.",
    "23 November 2011 - A Qazigund to Baramulla passenger derailed near Sadura in Anantnag district of Jammu and Kashmir injuring 20 people.",
    "11 January 2012 – The Brahmaputra Mail collided with a stationary freight train, killing 5 and injuring 9.",
    "26 February 2012 – The Trivandrum–Kozhikode Jan Shatabdi Express struck people who were standing on the track to watch fireworks, killing 3 and injuring 1.",
    "20 March 2012 – A train collided with an overcrowded taxi minivan at an unmanned railway crossing in northern Uttar Pradesh, 296 km (184 mi) from Lucknow, killing 15.",
    "26 March 2012 – A MEMU commuter train collided with a boulder-carrying truck at the Kannamangala gate on the outskirts of Bangalore, killing the pilot and driver.",
    "22 May 2012 – A cargo train and the Hubli-Bangalore Hampi Express collided in Penukonda, near Andhra Pradesh. The four bogies of the train derailed, one of them catching fire, and there were about 25 fatalities and about 43 injured.",
    "31 May 2012 – The Howrah–Dehradun Doon Express derailed near Jaunpur, killing at least 5 and injuring 15.",
    "19 July 2012 – The Vidarbha Express collided with a local train near Khardi station on the Mumbai–Kasara route, killing 1 and injuring 4.",
    "30 July 2012 – Tamil Nadu Express caught fire near Nellore in Andhra Pradesh, killing 47 and injuring 25.",
    "30 Nov 2012 – Grand Trunk Express caught fire to at least two coaches near Gwalior, killing several people.",
    "19 December 2012 – Indore–Yesvantpur Express collided with a truck near Sankhapur in Medak district, injuring many people.",
    "20 December 2012 – The locomotive of the Pune–Ernakulam Superfast Express derailed at Lanja village near Nivsar. There were no injuries.",
    "26 December 2012 – Head on collision between 55908 Dn Ledo – Dibrugarh Town passenger and Light Engine between Chabua and Panitola on New Tinsukia – Dibrugarh Town section of Tinsukia division of Northeast Frontier Railway. Injuring 71 persons.",
    "10 April 2013 – The 15228 Muzaffarpur–Yesvantpur Weekly Express derailed several carriages near Arakkonam, 40 km (25 mi) from Chennai, killing 1 and injuring 1.",
    "19 August 2013 – The Dhamara Ghat train accident occurred when the 12567 Saharsa–Patna Rajya Rani Superfast Express ran over passengers disembarking from another train at the Dhamara station in Bihar, killing 35.",
    "2 November 2013 – The 13352 Alappuzha–Dhanbad Express ran over passengers of the 57271 Vijayawada–Rayagada passenger train who had jumped onto the adjacent track due to a rumour that their train was on fire. 10 people were killed and at least 20 injured.",
    "13 November 2013 – The 2013 Chapramari Forest train accident occurred when a passenger train struck a herd of 40 elephants in the Chapramari Wildlife Sanctuary in West Bengal, killing 7 Indian elephants and injuring 10 others. The train was travelling at twice the speed limit.",
    "15 November 2013 – The 12618 Down Nizamuddin Ernakulam Mangala Lakshadweep Superfast Express had 13 coaches derail near Ghoti village, about 30 km (19 mi) from Nashik district, killing 3 or 4 and injuring dozens.",
    "28 December 2013 – The 16594 Bangalore City–Hazur Sahib Nanded Express caught fire near Kothacheruvu in Andhra Pradesh, killing at least 26 and injuring 12.",
    "20 March 2014 – A local train had six coaches uncouple and derail at Titwala, 61 km (38 mi) from Chhatrapati Shivaji Terminus, killing 1 and injuring 9.",
    "1 May 2014 – The 2014 Chennai train bombing occurred when two low-intensity bombs exploded on the Guwahati–Bangalore Kaziranga Express at Chennai Central railway station, killing 1 and injuring 14.",
    "4 May 2014 – The 50105 Diva Junction-Sawantvadi Passenger train derailed between Nagothane and Roha stations, killing about 20 and injuring 100.",
    "26 May 2014 – The 12556 Gorakhdham Express collided with a stationary freight train near Khalilabad station in Uttar Pradesh, killing at least 25 and injuring over 50.",
    "25 June 2014 – The 12236 Dibrugarh Rajdhani Express derailed near Chapra town, Bihar, killing 4 and injuring 8.",
    "23 July 2014 – Medak district bus-train collision; a Nanded-Secunderabad passenger train collided with a school bus at an unmanned level-crossing in Masaipet village of Medak district, killing 20.",
    "13 February 2015 – The Anekal derailment occurred when the 12677 Bangalore City–Ernakulam Intercity Express derailed nine coaches near Anekal in the Bangalore Urban district, killing 12 and injuring 100.",
    "20 March 2015 – The 2015 Uttar Pradesh train accident occurred when the Dehradun–Varanasi Janta Express derailed in Rae Bareli, Uttar Pradesh, killing 58 and injuring 150.",
    "25 May 2015 – The Muri Express derailed in Uttar Pradesh, killing 5 and injuring more than 50.",
    "4 August 2015 – The Harda twin train derailment occurred when the Kamayani Express and Janata Express derailed between Kurawan and Bhiringi stations in Madhya Pradesh, killing at least 31 people and injuring 100. Flash floods caused by Cyclonic Storm Komen dislodged a culvert causing a track misalignment, and several carriages of the 11071 Kamayani Express fell into the Machak river.",
    "4 September 2015 – 5 coaches including 3 AC coaches and an Unreserved coach along with SLR of 16859 DOWN Chennai Egmore – Mangalore Central Express derailed near Puvanur railway station, Villupuram – Vridhachalam section of Chord line. 39 passengers were reported injured.",
    "12 September 2015 – 9 coaches of the 12220 Secunderabad Junction–Lokmanya Tilak Terminus Duronto Express at Martur station, around 20 km (12 mi) from Kalaburagi town, Karnataka, killing 2 and injuring 7.",
    "12 September 2015 – A chartered Kalka–Shimla Shivalik Queen narrow-gauge railway derailed 3 coaches near Taksal, killing 2 and injuring 13. The train was boarded by a party of 36 customers and a tour operator, all from Britain. The Kalka-Shimla Railway is a part of the Mountain railways of India which has been enlisted as a UNESCO World Heritage Site since 2008.",
    "5 February 2016 – 4 coaches of the Kanyakumari-Bengaluru Island Express derailed near Patchur station. A few people were injured.",
    "6 May 2016 – Chennai Central–Thiruvananthapuram Central Superfast Express and a suburban train had a side collision near Pattabiram, injuring 7.",
    "27 August 2016 – Twelve coaches of the Thiruvananthapuram-Mangaluru Express derailed near Karukutty station on the Ernakulam – Thrissur section in Kerala. No casualties reported.",
    "29 September 2016 – A collision between Bhubaneswar-Bhadrak passenger train and a goods train at Cuttack, Odisha. One person killed and 22 injured.",
    "20 November 2016 – The Pukhrayan train derailment occurred when the 19321 Indore–Rajendra Nagar Express derailed 14 coaches at Pukhrayan, approximately 60 km (37 mi) from Kanpur, killing 152 and injuring 260.",
    "6 December 2016 – Rajendra Nagar-Guwahati Capital Express derailed at Samuktala Road station in Alipurduar district, West Bengal killing 2 people and injuring 6.",
    "28 December 2016 – Five coaches of Kurla-Ambarnath local train derailed in Mumbai. No injuries reported.",
    "28 December 2016 – 15 coaches of the Ajmer-Sealdah Express derailed near Rura station while crossing a bridge, injuring 44.",
    "21 January 2017 – The Kuneru train derailment occurred when the 18448 Jagdalpur–Bhubaneswar Hirakhand Express derailed near Kuneru, Vizianagaram, killing 41 and injuring 69.",
    "7 March 2017 – The 2017 Bhopal–Ujjain Passenger train bombing occurred when a bomb exploded on the Bhopal–Ujjain Passenger at Jabri railway station, injuring 10. This was the first strike in India by the Islamic State.",
    "30 March 2017 – Eight coaches of Mahakaushal Express derailed near Uttar Pradesh's Kulpahar, injuring 52.",
    "15 April 2017 – 8 coaches of the Meerut–Lucknow Rajya Rani Express derailed near Rampur; injuring at least 24.",
    "19 August 2017 – The 18478 Puri–Haridwar Kalinga Utkal Express derailed in Khatauli near Muzaffarnagar, Uttar Pradesh. Killing at least 23 and leaving around 97 injured.",
    "23 August 2017 – Auraiya train derailment occurred when the Kaifiyat Express (12225) derailed between Pata and Achalda railway stations around 02:40 am (IST). Around 100 people were injured.",
    "24 November 2017 – Vasco Da Gama-Patna Express derailment occurred in Chitrakoot, Uttar Pradesh killing three people and leaving around nine injured.",
    "24 November 2017 – Wagons of the Paradeep-Cuttack goods train loaded with imported coal derailed between Goraknath-Raghunathpur. No one was injured.",
    "25 April 2018 – Front wheels of WDP4D numbered 40405 which hauled 12606 UP Karaikkudi – Chennai Pallavan Superfast Express derailed while entering Tiruchchirappalli Junction railway station at 6:27 am. It was reported due to rail fracture, no injures and casualties reported. Then train was moved further with the delay of three hours towards Chennai.",
    "6 May 2018 – The WAP 4 locomotive of train 12810 Howrah-Mumbai Mail caught fire and resulted in death of assistant locomotive pilot and injury to locomotive pilot between Talni and Dhamangaon on Wardha – Badnera section. The enquiry stated that the fire was due to a defect in the locomotive.",
    "24 July 2018 – 5 killed, 4 injured in accident at St Thomas Mount station on Chennai Beach – Tambaram section. The victims were passengers on a Chennai Beach-Tirumalpur local who were hanging out from the doors when they were struck by a wall. This was unexpected because the train normally stopped at another platform but was diverted to this platform. 4 died on the spot.",
    "10 October 2018 – New Farakka Express accident: 7 Killed as Engine, 9 Coaches Derail in UP's Raebareli",
    "19 October 2018 – Amritsar train disaster: About 59 people were killed and about 100 injured when a train ran into a crowd of spectators who were standing on the tracks watching the Dusshera festival in Amritsar.",
    "3 February 2019 – Seemanchal Express derailment: 11 coaches derailed near Sahadai Buzurg railway station, about 50 km from Patna.",
    "31 March 2019 – 13 coaches of Tapti Ganga Express derailed near Gautamsthan railway station.",
    "20 April 2019 – 12 coaches of Poorva Express derailed in the outskirt of Kanpur Central near Rooma. No fatality was reported and 15 people were injured.",
    "29 August 2019 – Fire broke out in two coaches of Telangana Express at Asaoti railway station in Haryana.",
    "11 November 2019 – 16 passengers were injured and motorman killed when Lingampalli bound MMTS rammed into incoming Hundry Express at Kacheguda station. Hundry express had been given signal to enter the station when the outbound MMTS rammed into the train at the points.",
    "8 May 2020 – Aurangabad railway accident: 16 migrant workers sleeping on rail tracks were killed when a goods train ran over them between Jalna and Aurangabad districts.",
    "22 July 2020 – 3 railway employees from Hyderabad died after a double engine train ran them over. The incident took place between Vikarabad and Chittigadda railway station. The incident raised concerns about the safety of the railway employees and seems to be due to negligence from higher authorities.",
    "The parcel bogey of Malabar Express caught fire between Varkala and Paravur near Edavai railway station with no casualties.",
    "Two railway employees died after they were run over by a speeding express train when they were engaged in their routine track inspection work in Telangana's Mahabubabad district. They were working on track-1 when they noticed a train approaching on the track. They crossed to track-2 for safety. However, they failed to notice another train – Konark Superfast Express approaching on track-2 and were fatally hit by it.",
    "Two railway employees were fatally hit by a goods train while repairing a railway signal near Ambur railway station of Southern Railways. The reason for the accident stated by Railway Protection Force is that due to a heavy rain they couldn't hear the sound of the train at the same time and the locomotive pilot didn't have clear enough visibility to spot them.",
    "Four coaches including the pantry car of Guwahati-Howrah Saraighat COVID Special Train derailed at Chaygaon railway station on 25 August 2021.",
    "Coaches of the Bikaner–Guwahati Express derailed near New Domohani railway station in Mainaguri, Jalpaiguri at around 4:50 pm on 13 January 2022, causing 9 deaths.",
    "Pawan Express derails near Nashik on 3 April causing two injuries.",
    "25 March 2022 – Two trackmen were fatally hit by the 22209 Mumbai–New Delhi Duronto Express when they were engaged in their track inspection work between Navsari and Maroil Railway Stations; they failed to notice the approaching train and at the same time the locomotive pilot didn't have clear enough visibility due to heavy fog.",
    "2 January 2023 – 11 coaches of the Suryanagri Express derailed five minutes after it left Marwar Junction railway station at 3:27 a.m.. Ten people were injured.",
    "3 April 2023 – Eight passengers sustained burn injuries after Mohammad Shahrukh Saifi who was a resident of Delhi set fellow passengers on fire on board an express train in Elathur, Kozhikode, Kerala, which was bound for Kannur. Later in the incident, three people, including a two-year-old girl, were found dead on a railway track.",
    "15 May 2023 – One coach of the Chennai-Bangalore Double Decker Express heading towards Bengaluru derailed near Bisanattam station at around 11:30 am. No injuries or casualties reported.",
    "2 June 2023 – 2023 Odisha train collision: Train 12841 Coromandel Express running at 128 km/h (80 mph) collides with a freight train (goods train) loaded with iron ore in Odisha's Balasore district. The accident happened around 19:30 IST near Bahanaga Bazar station when the train was on the way to MGR Chennai Central from Shalimar railway station near Kolkata. More than 20 coaches were derailed. 12864 SMVT Bengaluru–Howrah SF Express travelling towards Howrah passed in the opposite line just seconds before at 130 km/h (81 mph). However, the derailed coaches of the Coromandel Express smashed into the last few coaches of the SMVT Bengaluru-Howrah SF Express before it could completely pass through that section. A total of three trains were involved. More than 1,200 were injured and 296 people died.",
    "8 Jun 2023 – The last coach of the 4-coach Nilgiri Mountain Railway heading from Ooty to Mettupalayam derailed near Coonoor station at around 3 pm. No injuries or casualties reported.",
    "9 Jun 2023 – A coach of the Vijayawada-Chennai Central Jan Shatabdi Express heading from Chennai Central back to shed derailed near Basin Bridge station at around 2:30 am. No injuries or casualties reported.",
    "11 Jun 2023 – The last coach of a Chennai suburban local train from Chennai Central towards Thiruvallur derailed near Basin Bridge station in the morning. No injuries or casualties reported.",
    "22 Jun 2023 – A coach of Lokmanya Tilak Express from Chennai to Mumbai caught fire near Vyasarpadi station due to a friction in a high-voltage power line. No injuries or casualties reported.",
    "25 Jun 2023 - A downward goods train collided at the back of another goods train at Ondagram railway station of Bankura district in West Bengal at around 4:00 AM (IST). The LP and ALP are both accused for breaking Signal.",
    "7 July 2023 – Three coaches in Howrah-bound Falaknuma Express caught fire between Bommaipally and Pagidipalli in Yadadri Bhuvanagiri district of Telangana. Reason for the horrific incident is unknown yet. No casualties or injuries have been reported.",
    "23 August 2023 - An under construction railway bridge over Kurung river on the Bairabi-Sairang line, in the mountainous area near Sairang around 21 kilometres from Aizawl in Mizoram, collapsed into the river killing at least 26 workers. All the fatalities were of migrant workers hailing from Malda district of West Bengal. Though a four-member investigation committee was formed, the findings are yet to be finalised.",
    "26 August 2023 - Around 5:15 AM (IST) fire erupted in Lucknow-Rameshwaram Bharat Gaurav train which was stationed near Madurai Junction; 9 killed, 20 injured. Preliminary investigation revealed that the passengers smuggled a gas cylinder aboard the train and were cooking in the coach when the fire broke out.",
    "23 September 2023 - The fire broke out in EOG & B1 coach of Tiruchirappalli - Shri Ganganagar Humsafar SF Express while crossing Valsad Railway Station",
    "26 September 2023 – On the night of 26 September 2023, and EMU train, which had originated from Shakur basti in Mathura, derailed and climbed onto platform number 2A at the station.",
    "11 October 2023 - On the night of 11 October 2023 at around 9:50 PM (local time) 6 coaches of 12506 Anand Vihar Terminal-Kamakhya Junction North East Express derailed near Raghunathpur Railway Station in Buxar district of Bihar killing 4 and injuring more than 70.",
    "29 October 2023 - On the evening of 29 October 2023 at around 9:02 PM (local time) a moving Visakhapatnam-Rayagada passenger train derailed after colliding with the Visakhapatnam-Palasa passenger train near Kottavalasa Junction railway station in Vizianagaram district, Andhra Pradesh killing at least 14 and injuring 50.",
    "On October 31, the Suheldev Superfast Express, on the route from Ghazipur City to Anand Vihar in Delhi, suffered a derailment incident in the outer region of Prayagraj, Uttar Pradesh. No casualties or injuries have been reported",
]

years = []
months = []
train_names = []
regions = []
reasons = []
deaths = []
injuries = []

for x in traindata:

    year = re.search(r"\d{4}",x)
    month = re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", x)
    death = re.search(r"(?:killing|deaths|killed|dead|deaths|kills|killing over|people|killing at least|killing more than) (\d+)",x)
    injury = re.search(r"(?:injured|injuring|Injuring|wounding|wounded|injures|injuring over|injuring more than) (\d+)",x)
    reason = re.search(r"\b(?:collided|collision|fire|derailed|terrorist|mob|attacked|hijacked|plunged|derail|derailment)\b",x)
    years.append(year.group() if year else None)
    months.append(month.group() if month else None)
    deaths.append(death.group(1) if death else None)
    injuries.append(injury.group(1) if injury else None)
    reasons.append(reason.group() if reason else None)

    doc = nlp(x)
    region = None
    train_name = None
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            region = ent.text.strip()
        elif "GPE" in ent.label_ or "Express" in ent.text or "Passenger" in ent.text or "passenger" in ent.text or "Train" in ent.text or "train" in ent.text:
            train_name = ent.text.strip()
    
    train_names.append(train_name)
    regions.append(region)

df = pd.DataFrame({
    'Years':years,
    'Months': months,
    'Train Name': train_names,
    'Reason': reasons,
    'Region': regions,
    'Deaths': deaths,
    'Injury': injuries,
})

print(df)

df.to_csv('D:/Mayank/Indian_rail_accident_1990-2023.csv',index=False)