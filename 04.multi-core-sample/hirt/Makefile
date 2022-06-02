include $(PWD)/../Makefile.param

FLAGS := -O3 -Wall -fPIC -fstack-protector -fstack-check -Wl,-z,relro,-z,now -fvisibility=hidden
export FLAGS

CPPFLAGS += -I$(REL_INC)
CPPFLAGS += -I$(PWD)/common/include
CPPFLAGS += -I$(PWD)/include
CPPFLAGS += $(FLAGS)
CPPFLAGS += -DON_BOARD
CPPFLAGS += $(LIBS_CFLAGS)

ifeq (${HIARCH}, hi3559av100)
CPPFLAGS += -DCHIP_3559A
endif

ifeq (${HIARCH}, hi3519av100)
CPPFLAGS += -DCHIP_3519A
endif

SRCS := $(wildcard src/*.c)
SRCS += $(wildcard common/src/*.c)
TARGET := sample_runtime_release

# target source
OBJS  := $(SRCS:%.c=%.o)

.PHONY : all proposal clean

all: proposal $(TARGET)


proposal:
	make -f Makefile.plugin.proposal
	make -f Makefile.plugin.proposal_without_permute
	@mkdir -p $(REL_LIB) && cp plugins/proposal/*.so  $(REL_LIB) -f
	@mkdir -p $(REL_LIB) && cp plugins/proposal_without_permute/*.so  $(REL_LIB) -f
	@cp plugins/proposal/*.so . -f
	@cp plugins/proposal_without_permute/*.so . -f

$(TARGET):$(OBJS)
	@cp $(PWD)/src/*.modelgroup $(PWD)
	$(CC) $(LIBS_LD_CFLAGS) -o $@ $^ -L$(REL_LIB) -Wl,-Bstatic -lsvpruntime -Wl,-Bdynamic -lpthread -lm -lstdc++ -L$(REL_LIB) -lnnie -ldl -lpthread -live -lsecurec -lmpi -ldnvqe -lupvqe -lVoiceEngine

clean:
	@rm -f $(TARGET)
	@rm -f $(OBJS)
	@rm -rf *.so
	@rm -rf $(PWD)/*.modelgroup
	make clean -f Makefile.plugin.proposal
	make clean -f Makefile.plugin.proposal_without_permute
