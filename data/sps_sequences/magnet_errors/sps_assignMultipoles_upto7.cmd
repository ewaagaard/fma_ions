
option,echo,warn;

AssignMultipoles : macro = {

// Dipole Errors
	EOPTION,SEED=98634628;
	Select, flag=error, clear = true;
//	Select,flag=error,class=RBEND, pattern="^MBA";
	Select,flag=error, class=multipole, pattern="^MBA";
	EFCOMP, dkn:={0,0,b3a+b3ar*tgauss(3),b4a+b4ar*tgauss(3),b5a+b5ar*tgauss(3),b6a,b7a+b7ar*tgauss(3),0,b9a,0,0,0,0,0,0,0};

	Select, flag=error, clear = true;
//	Select,flag=error,class=RBEND, pattern="^MBB";
	Select,flag=error,class=multipole, pattern="^MBB";
	EFCOMP, dkn:={0,0,b3b+b3br*tgauss(3),b4b+b4br*tgauss(3),b5b+b5br*tgauss(3),b6b,b7b+b7br*tgauss(3),0,b9b,0,0,0,0,0,0,0};

// Quadrupole Errors
	EOPTION,SEED=98634821;
	Select, flag=error, clear = true;
!	Select,flag=error,class=QUADRUPOLE, pattern="^QD";
	Select,flag=error,class=multipole, pattern="^QD";
	EFCOMP, dkn={0,0,0,b4d+b4dr*tgauss(3),0,b6d+b6dr*tgauss(3),0,b8d,0,0,0,0,0,0};

	Select, flag=error, clear = true;
!	Select,flag=error,class=QUADRUPOLE, pattern="^QF";
	Select,flag=error,class=multipole, pattern="^QF";
	EFCOMP, dkn={0,0,0,b4f+b4fr*tgauss(3),0,b6f+b6fr*tgauss(3),0,b8f,0,0,0,0,0,0};


// Sextupole Remanent fields
	Select, flag=error, clear = true;
	!Select,flag=error,class=SEXTUPOLE, pattern="^LSF";
	Select,flag=error,class=multipole, pattern="^LSF";
	EFCOMP, dkn={0,0,KSREMF+KSREM};

	Select, flag=error, clear = true;
	!Select,flag=error,class=SEXTUPOLE, pattern="^LSD";
	Select,flag=error,class=multipole, pattern="^LSD";
	EFCOMP, dkn={0,0,KSREMD-KSREM};

// Octupole Remanent fields
	Select, flag=error, clear = true;
	!Select,flag=error,class=OCTUPOLE, pattern="^LOF";
	Select,flag=error,class=multipole, pattern="^LOF";
	EFCOMP, dkn={0,0,0,KOREMF+KOREM};

	Select, flag=error, clear = true;
	!Select,flag=error,class=OCTUPOLE, pattern="^LOD";
	Select,flag=error,class=multipole, pattern="^LOD";
	EFCOMP, dkn={0,0,0,KOREMD-KOREM};

	SELECT,FLAG=ERROR,Full;
        ESAVE,FILE=err.out;
        READMYTABLE,file=err.out,table=errtab;


/*
// Misalign Dipoles
	EOPTION,SEED=123324,ADD=true;
	Select, flag=error, clear = true;
	Select,flag=error,class=RBEND;
	ealign,dx:=ON_misalign*tgauss(3)*0.5e-3,dy:=ON_misalign*tgauss(3)*0.5e-3,DS:=ON_misalign*tgauss(3)*1e-3,DPHI:=ON_tilt*tgauss(3)*0.3e-3,DTHETA:=ON_tilt*tgauss(3)*0.3e-3,DPSI:=ON_tilt*tgauss(3)*0.3e-3;

// Misalign Quads
	EOPTION,SEED=98634824,ADD=true;
	Select, flag=error, clear = true;
	Select,flag=error,class=QUADRUPOLE;
	ealign,dx:=ON_misalign*tgauss(3)*0.4e-3,dy:=ON_misalign*tgauss(3)*0.4e-3,DS:=ON_misalign*tgauss(3)*0.5e-3,DPHI:=ON_tilt*tgauss(3)*0.3e-3,DTHETA:=ON_tilt*tgauss(3)*0.3e-3,DPSI:=ON_tilt*tgauss(3)*0.3e-3;

// Misalign Sextupoles
	EOPTION,SEED=98634824,ADD=true;
	Select, flag=error, clear = true;
	Select,flag=error,class=SEXTUPOLE;
	ealign,dx:=ON_misalign*tgauss(3)*0.4e-3,dy:=ON_misalign*tgauss(3)*0.4e-3,DS:=ON_misalign*tgauss(3)*1e-3,DPHI:=ON_tilt*tgauss(3)*0.3e-3,DTHETA:=ON_tilt*tgauss(3)*0.3e-3,DPSI:=ON_tilt*tgauss(3)*0.3e-3;

// Errors in main components
	EOPTION,SEED=76182376,ADD=true;
	Select, flag=error, clear = true;
	Select,flag=error,class=RBEND;
	EFCOMP, dkn={ON_main*0.008445*tgauss(3)*5e-4,0,0,0,0,0,0,0,0,0,0,0};
	Select, flag=error, clear = true;
	Select,flag=error,class=QUADRUPOLE;
	EFCOMP, RADIUS=1,ORDER=1, dknr={0,ON_main*tgauss(3)*15e-4,0,0,0,0,0,0,0,0,0,0};


	SELECT,FLAG=ERROR,Full;
	ESAVE,FILE=err.out;


value, KQD, KQF1, b3a, b3b, b4a, b4b, b4f, b4fa, b4d, kLOF, kLOD, b5a, b5b, b6f, b6d, b7a, b7b;
fill, table=fittedmultipoles;
write, table=fittedmultipoles, file="Fittedmultipoles.txt";
*/
};
