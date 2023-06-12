# -*- coding: utf-8 -*-

import os
import hashlib
#######################################################################
# Arquivo de apoio para o Doc2VecRapido e o BertRapido 
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido
# Luiz Anísio 
# 10/03/2023 - disponibilizado no GitHub  
#######################################################################

#####################################
from multiprocessing.dummy import Pool as ThreadPool
class UtilDocs():
    
    @classmethod
    def hash(cls, texto):
        hash_object = hashlib.sha1(str(texto).encode())
        return f'{hash_object.hexdigest()}'

    @classmethod
    def progress_bar(cls, current_value, total, msg=''):
        increments = 25
        percentual = int((current_value / total) * 100)
        i = int(percentual // (100 / increments))
        text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
        print('{} {}           '.format(text, msg), end="\n" if percentual == 100 else "")

    @classmethod
    def map_thread(cls, func, lista, n_threads=5):
        # print('Iniciando {} threads'.format(n_threads))
        pool = ThreadPool(n_threads)
        pool.map(func, lista)
        pool.close()
        pool.join()  

    @classmethod
    def printlog(cls, inicio,msg, retornar_msg_erro=False, destaque = False):
        msg = f'> {inicio}: {msg}'
        if retornar_msg_erro:
            return msg
        if destaque: 
           cls.print_linha()
        print(msg)
        if destaque: 
           cls.print_linha()

    @classmethod
    def print_linha(cls, caractere='='):
        print(str(caractere * 70)[:70])

    # carrega os documentos txt de uma pasta e usa como tag o que estiver depois da palavra tags separadas por espaço
    # exemplo: 'arquivo texto 1 tags a b c.txt' tags será igual a ['a', 'b', 'c']
    @classmethod
    def carregar_documentos(cls, pasta, msg_log = 'UtilDocs'):
        cls.printlog(msg_log, f'Carregando documentos da pasta "{pasta}"')
        arquivos = cls.listar_arquivos(pasta)
        cls.printlog(msg_log, f'Documentos encontrados: {len(arquivos)}')
        documentos = []
        def _func(i):
            arquivo = arquivos[i]
            documento = cls.carregar_arquivo(arquivo)
            rotulo = os.path.split(arquivo)[1]            
            rotulo = os.path.splitext(rotulo)[0].lower()
            # se encontrar tag ou tags no nome do arquivo, busca as tags
            if rotulo.find('tags ') >=0:
               rotulos = rotulo.split('tags ')[1].strip().split(' ')
               print(f'Rótulo encontrado: {arquivo} >> tags: {rotulos}')
            elif rotulo.find('tag ') >=0:
               rotulos = rotulo.split('tag ')[1].strip().split(' ')
               print(f'Rótulo encontrado: {arquivo} >> tags: {rotulos}')
            else:
               rotulos = [rotulo.strip()]
            documentos.append( (documento, rotulos) )
        cls.map_thread(_func,list(range(len(arquivos))) )
        cls.printlog(msg_log, f'Documentos carregados: {len(documentos)}')
        #documentos, rotulos = list(zip(*documentos))
        #cls.printlog(f'Exemplo de documento: {documentos[0]} e rótulos: {rotulos[0]}')
        if len(documentos) == 0:
            return [],[]
        return list(zip(*documentos))

    # função simples de carga de arquivos que tenta descobrir o tipo de arquivo (utf8, ascii, latin1)
    @classmethod
    def carregar_arquivo(cls, arquivo):
        tipos = ['utf8', 'ascii', 'latin1']
        texto = None
        for tp in tipos:
            try:
                with open(arquivo, encoding=tp) as f:
                    texto = f.read()
                    break
            except UnicodeError:
                continue
        if not texto:
            with open(arquivo, encoding='latin1', errors='ignore') as f:
                texto = f.read()
        return texto.replace('<br>','\n') 

    @classmethod
    def listar_arquivos(cls, pasta, extensao='txt'):
        if not os.path.isdir(pasta):
            msg = f'Não foi encontrada a pasta "{pasta}" para listar os arquivos "{extensao}"'
            raise Exception(msg)
        res = []
        _extensao = f".{extensao}".lower() if extensao else ''
        for path, _, file_list in os.walk(pasta):
            for file_name in file_list:
                if file_name.lower().endswith(f"{_extensao}"):
                   res.append(os.path.join(path,file_name))
        return res

    @classmethod    
    def is_iterable(self, obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False 

    @classmethod
    def progresso_continuado(self, mensagem_inicial = None, i = None, total = None):
        '''Usar:
           progresso_continuado('Iniciando looping')
           progresso_continuado(i=10, total=100)
        '''
        if mensagem_inicial is None and i is None and total is None:
            return    
        if mensagem_inicial != None:
            print(f'{mensagem_inicial}', flush=True)
        if i is None:
            return
        if   total < 500:   salto = 10
        elif total < 1000:  salto = 100
        elif total < 10000: salto = 250
        else: salto = 500
        if i % salto == 0:
           print(f'{i}|', end = '', flush=True)   
        