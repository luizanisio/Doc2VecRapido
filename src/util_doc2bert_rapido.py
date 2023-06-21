# -*- coding: utf-8 -*-
#######################################################################
# Código para vetorizar documentos usando o Bert
#
# Bertimbau: https://huggingface.co/neuralmind/bert-base-portuguese-cased
#            https://github.com/neuralmind-ai/portuguese-bert/
#
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido/
# Luiz Anísio 
# 10/03/2023 - disponibilizado no GitHub  
#######################################################################

from sentence_transformers import SentenceTransformer
from util_doc2util import UtilDocs, Progresso
from scipy.spatial import distance
import numpy as np
import re
import os

# modelos:
#    '' ou l = neuralmind/bert-large-portuguese-cased
#    b = neuralmind/bert-base-portuguese-cased
#    [nome] para outros modelos

class Doc2BertRapido():
      ARQUIVO_MODELO = 'pytorch_model.bin'
      CARACTERES_POR_TOKEN = 3.75 # número mágico pela média de tamanho de tokens em um texto
      
      def __init__(self, modelo = 'l') -> None:
         if (not modelo.lower()) or (modelo.lower() in ('large','l','bert-large','bert_large')):
            if os.path.isdir('./bert-large-portuguese-cased'):
               self.modelo = './bert-large-portuguese-cased'
            else: 
               self.modelo = 'neuralmind/bert-large-portuguese-cased'
            self.nome_modelo = 'BERTimbau-large'
         elif modelo.lower() in ('b','base','bert-base','bert_base'):
            if os.path.isdir('./bert-base-portuguese-cased'):
               self.modelo = './bert-base-portuguese-cased'
            else: 
               self.modelo = 'neuralmind/bert-base-portuguese-cased'
            self.nome_modelo = 'BERTimbau-base'
         else:
            if not os.path.isdir(str(modelo)):
               self.printlog(f'carregnado modelo remoto "{modelo}"')
            self.modelo = modelo 
            self.nome_modelo = modelo

         self.printlog(f'Carregando modelo {self.nome_modelo}')
         self.model = SentenceTransformer(self.modelo)
         self.limite_tokens_modelo = int(self.model.get_max_seq_length()) 
         self.limite_caracteres_modelo = int(self.model.get_max_seq_length()) * self.CARACTERES_POR_TOKEN

      def vetor(self, texto, normalizar = True, retornar_tokens = False, retornar_float = True):
         vetores = self.model.encode([texto], normalize_embeddings=normalizar, show_progress_bar=False)
         vetor = vetores[0].tolist() if retornar_float else vetores[0]
         if not retornar_tokens:
            return vetor
         return vetor, self.tokens([texto])[0] 

      def similaridade(self, texto1, texto2):
         vetor1 = self.vetor_texto_grande(texto1, normalizar=False, retornar_float=False, retornar_tokens=False) 
         vetor2 = self.vetor_texto_grande(texto2, normalizar=False, retornar_float=False, retornar_tokens=False) 
         return 1- distance.cosine(vetor1,vetor2)   

      # todo: aqui os tetos grandes serão truncados
      # usar vetorizar_dados para tratamento especial
      def vetores(self, textos, normalizar = True, retornar_tokens = False, retornar_float = True, progresso=True):
         if len(textos) == 0:
            if retornar_tokens:
               return [], []
            return []
         vetores = self.model.encode(textos, normalize_embeddings=normalizar, show_progress_bar = (progresso and len(textos)>1))
         if retornar_float:
            vetores = vetores.tolist()
         if retornar_tokens:
            return vetores, self.tokens(textos) 
         return vetores

      def printlog(self,msg, retornar_msg_erro=False, destaque = False):
         UtilDocs.printlog('BertRapido', msg, retornar_msg_erro, destaque)

      def tokens(self, textos):
         inputs = self.model.tokenizer(textos, padding='longest', truncation=True, max_length = 1024)
         return [e.tokens for e in inputs.encodings]   

      def preprocess_br(self, texto):
         return self.tokens([texto])[0]

      def teste_modelo(self):
         #print('Init Sims', end='')
         #self.model.init_sims(replace=True)
         #print(' _o/')
         texto_1 = 'esse é um texto de teste para comparação - o teste depende de existirem os termos no vocab treinado'
         texto_2 = 'temos aqui um outro texto de teste para uma nova comparação mais distante - lembrando que o teste depende de existirem os termos no vocab treinado'
         texto_3 = 'esse é um texto de teste para comparação \n o teste depende de existirem os termos no vocab treinado'
         texto_4 = f'{texto_3}. ' * 200
         _texto_3 = texto_3.replace('\n',r'\n')
         texto_1_oov = texto_1.replace('texto','texto oovyyyyyyy oovxxxxxxxx').replace('teste', 'oovffffff teste oovssssss')
         UtilDocs.print_linha()
         print(' >>>> TESTE DO MODELO <<<<')
         UtilDocs.print_linha('- ')
         print('- Texto 01:  ', texto_1)
         print('  > tokens: ', '|'.join(self.preprocess_br(texto_1)) )
         print('- Texto 02:  ', texto_2)
         print('  > tokens: ', '|'.join(self.preprocess_br(texto_2)) )
         print('- Texto 03:  ', _texto_3)
         print('  > tokens: ', '|'.join(self.preprocess_br(texto_3)) )
         print('- Texto 04:  200 x Texto 03')
         UtilDocs.print_linha('- ')
         sim = 100*self.similaridade(texto_1, texto_1)
         print(f'Similaridade entre o texto 01 e ele mesmo: {sim:.2f}%')          
         sim = 100*self.similaridade(texto_1, texto_1_oov)
         print(f'Similaridade entre o texto 01 e ele com oov: {sim:.2f}%')          
         sim = 100*self.similaridade(texto_1, texto_2)
         print(f'Similaridade entre os textos 01 e 02: {sim:.2f}%')          
         sim = 100*self.similaridade(texto_1, texto_3)
         print(f'Similaridade entre os textos 01 e 03: {sim:.2f}%')          
         sim = 100*self.similaridade(texto_3, texto_4)
         print(f'Similaridade entre os textos 03 e 04 (grande): {sim:.2f}%')          
         UtilDocs.print_linha()

      def vetorizar_dados(self, dados, epocas = None):
         _pos = []
         _textos = []
         # verifica dados que não possuem vetor (pode ter vindo do cache ou trazidos da base de dados)
         self.printlog('Analisando textos grandes e dados já vetorizados')
         #qtd = len(dados)
         qtd_grandes, qtd_vazios, qtd_ok = [0], [0], [0]
         print(f'Vetorizando {len(dados)} dados com Doc2Bert')
         print('Vetorizando textos longos...')
         bar = Progresso(total = len(dados))
         restam = [len(dados)]
         #for i in range(len(dados)):
         def __vetorizar__(i):
             #UtilDocs.progress_bar(i, qtd,f' vetorizando textos grandes {i+1}/{qtd} textos                ')
             restam[0] -= 1
             bar.restantes(restam[0])
             if 'vetor' in dados[i] and type(dados[i]['vetor']) in (list, tuple):
                # já tem vetor, verifica o hash ou cria um único
                qtd_ok[0] += 1
                return
             
             _texto = dados[i].get('texto','')
             # sem texto
             if not _texto:
                dados[i]['texto'] = ''
                dados[i]['hash'] = 'vazio'
                dados[i]['vetor'] = None
                qtd_vazios[0] += 1
                return 
             # textos grandes, tem que vetorizar o batch para cada texto com seus pedacos
             if len(_texto) > self.limite_caracteres_modelo:
                vetor = self.vetor_texto_grande(_texto, retornar_float=True, retornar_tokens=False)
                dados[i]['vetor'] = vetor
                dados[i]['hash'] = dados[i].get('hash', UtilDocs.hash(_texto))
                qtd_grandes[0] += 1
                return
             # pilha de vetorização 
             _pos.append(i)
             _textos.append(dados[i].get('texto',''))
         UtilDocs.map_thread(__vetorizar__, range(len(dados)), n_threads=UtilDocs.N_THREADS_PADRAO, tdqm_progress=True)    
         #UtilDocs.progress_bar(1,1,' Concluído   
         bar.restantes(0)
         bar.close()
         self.printlog(f'Vetorizados {qtd_grandes[0]} textos grandes, {qtd_vazios[0]} documentos estão vazios e {qtd_ok[0]} já possuem vetores                   ')
         self.printlog(f'Vetorizando {len(_textos)} textos pequenos')
         vetores = self.vetores(_textos, normalizar=True, retornar_float=True, retornar_tokens=False)
         self.printlog('Consolidadno vetorização ...')
         for i, pos in enumerate(_pos):
               dados[pos]['vetor'] = vetores[i]
               dados[pos]['hash'] = UtilDocs.hash(_textos[i])

      RE_QUEBRA_GRANDES = re.compile('[^!?。.？！]+[!?。.？！]?')
      RE_QUEBRA_PEQUENAS = re.compile('[^,;]+[,;]?')
      def vetor_texto_grande(self, texto, retornar_tokens = True, normalizar = True, retornar_float = True):
          if len(texto) <= self.limite_caracteres_modelo:
             return self.vetor(texto, normalizar=normalizar, retornar_tokens=retornar_tokens, retornar_float = retornar_float)
          primeiras_quebras = self.RE_QUEBRA_GRANDES.findall(texto)
          quebras = []
          # caso alguma quebra ainda seja muito grande, quebra por outros separadores
          for quebra in primeiras_quebras:
             if len(quebra) > self.limite_caracteres_modelo:
                novas = self.RE_QUEBRA_PEQUENAS.findall(quebra)
                quebras += novas
             else:
                quebras.append(quebra)
          pedacos = []
          anterior = ''
          # une as quebras até o limite     
          for pedaco in quebras:
             if len(anterior) + len(pedaco) > self.limite_caracteres_modelo:
                if anterior:
                   pedacos.append(anterior)
                anterior = pedaco
             else:
                anterior += f' {pedaco}'
          if anterior:
             pedacos.append(anterior)
          # para o caso de retornar só o vetor, calcula a média dos vetores
          if not retornar_tokens:
             vetores = self.vetores(pedacos, normalizar=normalizar, retornar_tokens=False, retornar_float = False, progresso = False)    
             vetor = np.array(vetores).mean(axis=0)
             return vetor.tolist() if retornar_float else vetor
          # une os tokens e tira a média dos vetores
          vetores, tokens_pedacos = self.vetores(pedacos, normalizar=normalizar, retornar_tokens=True, retornar_float = False)    
          tokens = []
          for _ in tokens_pedacos:
             tokens += _
          vetor = np.array(vetores).mean(axis=0)
          if retornar_float:
             vetor = vetor.tolist()
          return vetor, tokens

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-modelo', help='modelo padrao large', required=False)
    args = parser.parse_args()

    TIPO_MODELO = args.modelo or ''

    db = Doc2BertRapido(TIPO_MODELO)
    db.teste_modelo()

