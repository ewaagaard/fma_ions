SPS_matchtunes(Qx,Qy) : macro = {
 MATCH, SEQUENCE=SPS;
	VARY, NAME=KQD,  STEP=1.E-8;
	VARY, NAME=KQF1,  STEP=1.E-8;
	GLOBAL, Q1=Qx, Q2=Qy;
	LMDIF, CALLS=1000, TOLERANCE=1.E-22;
 ENDMATCH;
};


SPS_matchfractionaltunes_Q20(Qx,Qy) : macro = {
 MATCH, SEQUENCE=SPS;
	VARY, NAME=KQD,  STEP=1.E-8;
	VARY, NAME=KQF1,  STEP=1.E-8;
	GLOBAL, Q1=20+Qx, Q2=20+Qy;
	LMDIF, CALLS=1000, TOLERANCE=1.E-22;
 ENDMATCH;
};


SPS_matchfractionaltunes_Q26(Qx,Qy) : macro = {
 MATCH, SEQUENCE=SPS;
	VARY, NAME=KQD,  STEP=1.E-8;
	VARY, NAME=KQF1,  STEP=1.E-8;
	GLOBAL, Q1=26+Qx, Q2=26+Qy;
	LMDIF, CALLS=1000, TOLERANCE=1.E-22;
 ENDMATCH;
};

SPS_matchfractionaltunes_Q22(Qx,Qy) : macro = {
 MATCH, SEQUENCE=SPS;
	VARY, NAME=KQD,  STEP=1.E-8;
	VARY, NAME=KQF1,  STEP=1.E-8;
	GLOBAL, Q1=22+Qx, Q2=22+Qy;
	LMDIF, CALLS=1000, TOLERANCE=1.E-22;
 ENDMATCH;
};

SPS_setfractionaltunes_Q20(Qx, Qy): macro = {
	option, -info;
	dkqd_h =    -0.000116739321 ;
	dkqf1_h =    0.0003911951437 ;
	dkqd_v =   -0.0003920071709 ;
	dkqf1_v =    0.0001170949059 ;

	KQF1 :=  0.01157957644 + (Qx - 0.13) * dkqf1_h + (Qy - 0.18) * dkqf1_v;
	KQD  := -0.01158097147 + (Qx - 0.13) * dkqd_h  + (Qy - 0.18) * dkqd_v;
	option, info;
};


SPS_setfractionaltunes_Q26(Qx, Qy): macro = {
	option, -info;
	dkqd_h =    -7.127348119e-05 ;
	dkqf1_h =    0.0003701440969 ;
	dkqd_v =   -0.0003705446173 ;
	dkqf1_v =    7.147226908e-05 ;

	KQF1 :=  0.01443603711 + (Qx - 0.13) * dkqf1_h + (Qy - 0.18) * dkqf1_v;
	KQD  := -0.01443946072 + (Qx - 0.13) * dkqd_h  + (Qy - 0.18) * dkqd_v;
	option, info;
};


SPS_setfractionaltunes_Q22(Qx, Qy): macro = {
	option, -info;
	dkqd_h =    -0.0001004546891 ;
	dkqf1_h =    0.0003869133922 ;
	dkqd_v =   -0.0003875753452 ;
	dkqf1_v =    0.0001007449219 ;

	KQF1 :=  0.01257544512 + (Qx - 0.13) * dkqf1_h + (Qy - 0.18) * dkqf1_v;
	KQD  := -0.01257768331 + (Qx - 0.13) * dkqd_h  + (Qy - 0.18) * dkqd_v;
	option, info;
};    


SPS_setchroma_Q20(QPH_setvalue, QPV_setvalue) : macro = {
	option, -info;
	LSDA0=-0.040896;
	LSDB0=-0.063333733;	
	LSFA0=0.04516855;
	LSFB0=0.026760516;
	LSFC0=0.04516855;

	logical.LSDAQPH=-0.0090643256;	 
	logical.LSDBQPH=-0.0140406480;
	logical.LSFAQPH=0.0300341594;
	logical.LSFBQPH=0.0177853536;
	logical.LSFCQPH=0.0300341594;
	
	logical.LSDAQPV=-0.0272610246;
	logical.LSDBQPV=-0.0422147313;
	logical.LSFAQPV=0.0100859900;
	logical.LSFBQPV=0.0059841919;
	logical.LSFCQPV=0.0100859900;
		
	ksda:=logical.LSDAQPH*QPH_setvalue+logical.LSDAQPV*QPV_setvalue+LSDA0;
	ksdb:=logical.LSDBQPH*QPH_setvalue+logical.LSDBQPV*QPV_setvalue+LSDB0;
	ksfa:=logical.LSFAQPH*QPH_setvalue+logical.LSFAQPV*QPV_setvalue+LSFA0;
	ksfb:=logical.LSFBQPH*QPH_setvalue+logical.LSFBQPV*QPV_setvalue+LSFB0;
	ksfc:=logical.LSFCQPH*QPH_setvalue+logical.LSFCQPV*QPV_setvalue+LSFC0;

	kLSDA := ksda;
	kLSDB := ksdb;
	kLSFA := ksfa;
	kLSFB := ksfb;
	kLSFC := ksfc;	
	option, info;
};


SPS_setchroma_Q26(QPH_setvalue, QPV_setvalue) : macro = {
	option, -info;
	LSDA0=-0.149628261;
	LSDB0=-0.145613183;	
	LSFA0=0.063256459;
	LSFB0=0.121416689;
	LSFC0=0.063256459;

	logical.LSDAQPH=.011283;	 
	logical.LSDBQPH=-.040346;
	logical.LSFAQPH=.04135;
	logical.LSFBQPH=.079565;
	logical.LSFCQPH=.04135;
	
	logical.LSDAQPV=-.11422;
	logical.LSDBQPV=-.08606;
	logical.LSFAQPV=.0097365;
	logical.LSFBQPV=.016931;
	logical.LSFCQPV=.0097365;
		
	ksda:=logical.LSDAQPH*QPH_setvalue+logical.LSDAQPV*QPV_setvalue+LSDA0;
	ksdb:=logical.LSDBQPH*QPH_setvalue+logical.LSDBQPV*QPV_setvalue+LSDB0;
	ksfa:=logical.LSFAQPH*QPH_setvalue+logical.LSFAQPV*QPV_setvalue+LSFA0;
	ksfb:=logical.LSFBQPH*QPH_setvalue+logical.LSFBQPV*QPV_setvalue+LSFB0;
	ksfc:=logical.LSFCQPH*QPH_setvalue+logical.LSFCQPV*QPV_setvalue+LSFC0;
	
	kLSDA := ksda;
	kLSDB := ksdb;
	kLSFA := ksfa;
	kLSFB := ksfb;
	kLSFC := ksfc;	
	option, info;
};


SPS_setchroma_Q22(QPH_setvalue, QPV_setvalue) : macro = {
	option, -info;
	lsda0 =         -0.0268449 ;
	lsdb0 =          -0.118119 ;
	lsfa0 =          0.0594952 ;
	lsfb0 =          0.0429914 ;
	lsfc0 =          0.0594952 ;

	logical.lsdaqph =     -0.01156955797 ;
	logical.lsdbqph =     -0.01925665038 ;
	logical.lsfaqph =      0.04232697265 ;
	logical.lsfbqph =      0.02370989653 ;
	logical.lsfcqph =      0.04232697265 ;

	logical.lsdaqpv =     -0.03951301487 ;
	logical.lsdbqpv =      -0.0657585618 ;
	logical.lsfaqpv =      0.01250092378 ;
	logical.lsfbqpv =     0.007011866493 ;
	logical.lsfcqpv =      0.01250092378 ;
		
	ksda:=logical.LSDAQPH*QPH_setvalue+logical.LSDAQPV*QPV_setvalue+LSDA0;
	ksdb:=logical.LSDBQPH*QPH_setvalue+logical.LSDBQPV*QPV_setvalue+LSDB0;
	ksfa:=logical.LSFAQPH*QPH_setvalue+logical.LSFAQPV*QPV_setvalue+LSFA0;
	ksfb:=logical.LSFBQPH*QPH_setvalue+logical.LSFBQPV*QPV_setvalue+LSFB0;
	ksfc:=logical.LSFCQPH*QPH_setvalue+logical.LSFCQPV*QPV_setvalue+LSFC0;

	kLSDA := ksda;
	kLSDB := ksdb;
	kLSFA := ksfa;
	kLSFB := ksfb;
	kLSFC := ksfc;	
	option, info;
};
