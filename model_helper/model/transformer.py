# from keras import Input, Model
# from keras.layers import Dense, Dropout, Embedding
#
#
# def transformer_model(input_vocab_size, output_vocab_size, max_seq_len, d_model, num_heads, d_ff, dropout_rate):
#     # Encoder
#     encoder_inputs = Input(shape=(max_seq_len,))
#     enc_emb = Embedding(input_vocab_size, d_model)(encoder_inputs)
#     enc_pos_emb = PositionalEncoding(max_seq_len, d_model)(enc_emb)
#     enc_dropout = Dropout(dropout_rate)(enc_pos_emb)
#     encoder_outputs = TransformerEncoder(num_heads, d_ff, d_model, dropout_rate)(enc_dropout)
#
#     # Decoder
#     decoder_inputs = Input(shape=(max_seq_len,))
#     dec_emb = Embedding(output_vocab_size, d_model)(decoder_inputs)
#     dec_pos_emb = PositionalEncoding(max_seq_len, d_model)(dec_emb)
#     dec_dropout = Dropout(dropout_rate)(dec_pos_emb)
#     decoder_outputs = TransformerDecoder(num_heads, d_ff, d_model, dropout_rate)(dec_dropout, encoder_outputs)
#
#     # Output
#     outputs = Dense(output_vocab_size, activation='softmax')(decoder_outputs)
#
#     # Define MODEL
#     model = Model([encoder_inputs, decoder_inputs], outputs)
#
#     return model