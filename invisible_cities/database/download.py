import sqlite3
import pymysql
import pymysql as MySQLdb
pymysql.install_as_MySQLdb()
import os
from os import path
from base64 import b64decode as dec


def loadDB():
    dbfile = path.join(os.environ['ICDIR'], 'database/localdb.sqlite3')
    try:
        os.remove(dbfile)
    except:
        pass

    connSql3 = sqlite3.connect(dbfile)
    cursorSql3 = connSql3.cursor()

    connMySql = MySQLdb.connect(host="neutrinos1.ific.uv.es", user=dec('am1iZW5sbG9jaA=='),
                                passwd=eval(dec('Jycuam9pbihtYXAobGFtYmRhIGM6IGNocihjLTUpLCBbNzIsIDEwMiwgMTE1LCAxMDcsIDExOSwgMTAyLCAxMTUsIDEwNF0pKQ==')),
                                db="ICNEWDB")
    cursorMySql = connMySql.cursor()

    # Create tables
    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `DetectorGeo` (
  `XMIN` float NOT NULL
,  `XMAX` float NOT NULL
,  `YMIN` float NOT NULL
,  `YMAX` float NOT NULL
,  `ZMIN` float NOT NULL
,  `ZMAX` float NOT NULL
,  `RMAX` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtBlr` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `coeff_c` double NOT NULL
,  `coeff_blr` double NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtGain` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `adc_to_pes` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtSigma` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `sigma` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtMapping` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `ChannelID` integer NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtMask` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `Active` integer NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtNoiseRms` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `noise_rms` double NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtPosition` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `PmtID` varchar(5) NOT NULL
,  `X` float NOT NULL
,  `Y` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmBaseline` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `Energy` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmGain` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `adc_to_pes` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmMapping` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `ChannelID` integer NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmMask` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `Active` integer NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmNoise` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `0` float NOT NULL
,  `1` float NOT NULL
,  `2` float NOT NULL
,  `3` float NOT NULL
,  `4` float NOT NULL
,  `5` float NOT NULL
,  `6` float NOT NULL
,  `7` float NOT NULL
,  `8` float NOT NULL
,  `9` float NOT NULL
,  `10` float NOT NULL
,  `11` float NOT NULL
,  `12` float NOT NULL
,  `13` float NOT NULL
,  `14` float NOT NULL
,  `15` float NOT NULL
,  `16` float NOT NULL
,  `17` float NOT NULL
,  `18` float NOT NULL
,  `19` float NOT NULL
,  `20` float NOT NULL
,  `21` float NOT NULL
,  `22` float NOT NULL
,  `23` float NOT NULL
,  `24` float NOT NULL
,  `25` float NOT NULL
,  `26` float NOT NULL
,  `27` float NOT NULL
,  `28` float NOT NULL
,  `29` float NOT NULL
,  `30` float NOT NULL
,  `31` float NOT NULL
,  `32` float NOT NULL
,  `33` float NOT NULL
,  `34` float NOT NULL
,  `35` float NOT NULL
,  `36` float NOT NULL
,  `37` float NOT NULL
,  `38` float NOT NULL
,  `39` float NOT NULL
,  `40` float NOT NULL
,  `41` float NOT NULL
,  `42` float NOT NULL
,  `43` float NOT NULL
,  `44` float NOT NULL
,  `45` float NOT NULL
,  `46` float NOT NULL
,  `47` float NOT NULL
,  `48` float NOT NULL
,  `49` float NOT NULL
,  `50` float NOT NULL
,  `51` float NOT NULL
,  `52` float NOT NULL
,  `53` float NOT NULL
,  `54` float NOT NULL
,  `55` float NOT NULL
,  `56` float NOT NULL
,  `57` float NOT NULL
,  `58` float NOT NULL
,  `59` float NOT NULL
,  `60` float NOT NULL
,  `61` float NOT NULL
,  `62` float NOT NULL
,  `63` float NOT NULL
,  `64` float NOT NULL
,  `65` float NOT NULL
,  `66` float NOT NULL
,  `67` float NOT NULL
,  `68` float NOT NULL
,  `69` float NOT NULL
,  `70` float NOT NULL
,  `71` float NOT NULL
,  `72` float NOT NULL
,  `73` float NOT NULL
,  `74` float NOT NULL
,  `75` float NOT NULL
,  `76` float NOT NULL
,  `77` float NOT NULL
,  `78` float NOT NULL
,  `79` float NOT NULL
,  `80` float NOT NULL
,  `81` float NOT NULL
,  `82` float NOT NULL
,  `83` float NOT NULL
,  `84` float NOT NULL
,  `85` float NOT NULL
,  `86` float NOT NULL
,  `87` float NOT NULL
,  `88` float NOT NULL
,  `89` float NOT NULL
,  `90` float NOT NULL
,  `91` float NOT NULL
,  `92` float NOT NULL
,  `93` float NOT NULL
,  `94` float NOT NULL
,  `95` float NOT NULL
,  `96` float NOT NULL
,  `97` float NOT NULL
,  `98` float NOT NULL
,  `99` float NOT NULL
,  `100` float NOT NULL
,  `101` float NOT NULL
,  `102` float NOT NULL
,  `103` float NOT NULL
,  `104` float NOT NULL
,  `105` float NOT NULL
,  `106` float NOT NULL
,  `107` float NOT NULL
,  `108` float NOT NULL
,  `109` float NOT NULL
,  `110` float NOT NULL
,  `111` float NOT NULL
,  `112` float NOT NULL
,  `113` float NOT NULL
,  `114` float NOT NULL
,  `115` float NOT NULL
,  `116` float NOT NULL
,  `117` float NOT NULL
,  `118` float NOT NULL
,  `119` float NOT NULL
,  `120` float NOT NULL
,  `121` float NOT NULL
,  `122` float NOT NULL
,  `123` float NOT NULL
,  `124` float NOT NULL
,  `125` float NOT NULL
,  `126` float NOT NULL
,  `127` float NOT NULL
,  `128` float NOT NULL
,  `129` float NOT NULL
,  `130` float NOT NULL
,  `131` float NOT NULL
,  `132` float NOT NULL
,  `133` float NOT NULL
,  `134` float NOT NULL
,  `135` float NOT NULL
,  `136` float NOT NULL
,  `137` float NOT NULL
,  `138` float NOT NULL
,  `139` float NOT NULL
,  `140` float NOT NULL
,  `141` float NOT NULL
,  `142` float NOT NULL
,  `143` float NOT NULL
,  `144` float NOT NULL
,  `145` float NOT NULL
,  `146` float NOT NULL
,  `147` float NOT NULL
,  `148` float NOT NULL
,  `149` float NOT NULL
,  `150` float NOT NULL
,  `151` float NOT NULL
,  `152` float NOT NULL
,  `153` float NOT NULL
,  `154` float NOT NULL
,  `155` float NOT NULL
,  `156` float NOT NULL
,  `157` float NOT NULL
,  `158` float NOT NULL
,  `159` float NOT NULL
,  `160` float NOT NULL
,  `161` float NOT NULL
,  `162` float NOT NULL
,  `163` float NOT NULL
,  `164` float NOT NULL
,  `165` float NOT NULL
,  `166` float NOT NULL
,  `167` float NOT NULL
,  `168` float NOT NULL
,  `169` float NOT NULL
,  `170` float NOT NULL
,  `171` float NOT NULL
,  `172` float NOT NULL
,  `173` float NOT NULL
,  `174` float NOT NULL
,  `175` float NOT NULL
,  `176` float NOT NULL
,  `177` float NOT NULL
,  `178` float NOT NULL
,  `179` float NOT NULL
,  `180` float NOT NULL
,  `181` float NOT NULL
,  `182` float NOT NULL
,  `183` float NOT NULL
,  `184` float NOT NULL
,  `185` float NOT NULL
,  `186` float NOT NULL
,  `187` float NOT NULL
,  `188` float NOT NULL
,  `189` float NOT NULL
,  `190` float NOT NULL
,  `191` float NOT NULL
,  `192` float NOT NULL
,  `193` float NOT NULL
,  `194` float NOT NULL
,  `195` float NOT NULL
,  `196` float NOT NULL
,  `197` float NOT NULL
,  `198` float NOT NULL
,  `199` float NOT NULL
,  `200` float NOT NULL
,  `201` float NOT NULL
,  `202` float NOT NULL
,  `203` float NOT NULL
,  `204` float NOT NULL
,  `205` float NOT NULL
,  `206` float NOT NULL
,  `207` float NOT NULL
,  `208` float NOT NULL
,  `209` float NOT NULL
,  `210` float NOT NULL
,  `211` float NOT NULL
,  `212` float NOT NULL
,  `213` float NOT NULL
,  `214` float NOT NULL
,  `215` float NOT NULL
,  `216` float NOT NULL
,  `217` float NOT NULL
,  `218` float NOT NULL
,  `219` float NOT NULL
,  `220` float NOT NULL
,  `221` float NOT NULL
,  `222` float NOT NULL
,  `223` float NOT NULL
,  `224` float NOT NULL
,  `225` float NOT NULL
,  `226` float NOT NULL
,  `227` float NOT NULL
,  `228` float NOT NULL
,  `229` float NOT NULL
,  `230` float NOT NULL
,  `231` float NOT NULL
,  `232` float NOT NULL
,  `233` float NOT NULL
,  `234` float NOT NULL
,  `235` float NOT NULL
,  `236` float NOT NULL
,  `237` float NOT NULL
,  `238` float NOT NULL
,  `239` float NOT NULL
,  `240` float NOT NULL
,  `241` float NOT NULL
,  `242` float NOT NULL
,  `243` float NOT NULL
,  `244` float NOT NULL
,  `245` float NOT NULL
,  `246` float NOT NULL
,  `247` float NOT NULL
,  `248` float NOT NULL
,  `249` float NOT NULL
,  `250` float NOT NULL
,  `251` float NOT NULL
,  `252` float NOT NULL
,  `253` float NOT NULL
,  `254` float NOT NULL
,  `255` float NOT NULL
,  `256` float NOT NULL
,  `257` float NOT NULL
,  `258` float NOT NULL
,  `259` float NOT NULL
,  `260` float NOT NULL
,  `261` float NOT NULL
,  `262` float NOT NULL
,  `263` float NOT NULL
,  `264` float NOT NULL
,  `265` float NOT NULL
,  `266` float NOT NULL
,  `267` float NOT NULL
,  `268` float NOT NULL
,  `269` float NOT NULL
,  `270` float NOT NULL
,  `271` float NOT NULL
,  `272` float NOT NULL
,  `273` float NOT NULL
,  `274` float NOT NULL
,  `275` float NOT NULL
,  `276` float NOT NULL
,  `277` float NOT NULL
,  `278` float NOT NULL
,  `279` float NOT NULL
,  `280` float NOT NULL
,  `281` float NOT NULL
,  `282` float NOT NULL
,  `283` float NOT NULL
,  `284` float NOT NULL
,  `285` float NOT NULL
,  `286` float NOT NULL
,  `287` float NOT NULL
,  `288` float NOT NULL
,  `289` float NOT NULL
,  `290` float NOT NULL
,  `291` float NOT NULL
,  `292` float NOT NULL
,  `293` float NOT NULL
,  `294` float NOT NULL
,  `295` float NOT NULL
,  `296` float NOT NULL
,  `297` float NOT NULL
,  `298` float NOT NULL
,  `299` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmNoiseBins` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `Bin` integer NOT NULL
,  `Energy` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmPosition` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `X` float NOT NULL
,  `Y` float NOT NULL
);''')

    tables = ['DetectorGeo','PmtBlr','PmtGain','PmtMapping','PmtMask',
          'PmtNoiseRms','PmtPosition','PmtSigma','SipmBaseline','SipmGain',
          'SipmMapping','SipmMask','SipmNoise','SipmNoiseBins','SipmPosition']


    # Copy all tables
    for table in tables:
        # Get all data
        cursorMySql.execute('SELECT * from {0}'.format(table))
        data = cursorMySql.fetchall()

        # Insert all rows
        fields = '?'
        nfields = len(data[0])
        fields += (nfields-1) * ',?'
        cursorSql3.executemany('INSERT INTO {0} VALUES({1})'.format(table,fields),data)
        connSql3.commit()


if __name__ == '__main__':
    loadDB()
