#
# Make file for Face Detector
#
# Author: LI XUANPENG <marshallixp@gmail.com>
# Date: 15/02/2014

OCVFLAGS 	= `pkg-config --cflags opencv`
OCVLIBS 	= `pkg-config --libs opencv` -lstdc++

CPP 		= g++

# type of system?
LBITS 		= $(shell /usr/bin/getconf LONG_BIT)
ifeq ($(LBITS),64)
# do 64 bit stuff here, like set some CFLAGS
	SFX = 64
else
	SFX = 32
endif

DIR 		= ./FusionCenter
DIR_SIM		= ./RunFusionCenter

SRCS		+= $(wildcard ${DIR_SIM}/*.cpp ${DIR}/*.cpp) 

OBJS  		= $(patsubst %.cpp, %.o, $(SRCS))

CFLAGS		+= $(OCVFLAGS)
LIBS		+= $(OCVLIBS)
BINARY		= FusionSystem$(SFX)

all: release

release: $(OBJS)
		$(CPP) $^ $(LDFLAGS) $(LIBS) $(CFLAGS) -o $(BINARY)

.cpp.o:
		$(CPP) $< $(CFLAGS) $(LIBS) $(LDFLAGS) -c -o $@

clean: 
		rm -f $(DIR)/*.o  $(DIR_SIM)/*.o $(BINARY)
