#WARNING: Don't invoke this file independently, must be included into parent folder makefile

# testing_prec_def.h
C_SRC_BATCH=
C_SRC_BATCH+=		\
	test_dvecchia_batch.c
#	test_svecchia_batch.c	\
#	test_zvecchia_batch.c	\
#	test_cvecchia_batch.c

$(OBJ_DIR)/test_%vecchia_batch.o: $(VECCHIA_SOURCE)/test_Xvecchia_batch.cpp
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(BIN_DIR)/test_%_batch: $(OBJ_DIR)/test_%_batch.o
	$(CXX) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)

EXE_BATCH= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC_BATCH))
