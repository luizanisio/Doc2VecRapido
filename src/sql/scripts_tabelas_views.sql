/* database de testes */
create database testes;

/* tabela de vetores */
CREATE TABLE testes.vetores (
    seq_documento int(11) NOT NULL COMMENT 'seq_documento' ,
    pagina int(11) NOT NULL COMMENT 'pagina do documento' ,
    vetor blob NOT NULL COMMENT 'vetor criado por algum modelo e inserido no padrão hexadecimal do memsql',
    dthr TIMESTAMP default now() COMMENT 'data de inclusão do vetor na tabela',
    PRIMARY KEY (seq_documento),
    SHARD KEY (seq_documento)
);

/* view consulta similares */
create VIEW testes.vw_similares as
    select v1.seq_documento as seq_documento_1, v2.seq_documento as seq_documento_2, v1.pagina as pagina_1, v2.pagina as pagina_2,
           v1.dthr as dthr_1, v2.dthr as dthr_2, dot_product(v1.vetor,v2.vetor) as sim, round(dot_product(v1.vetor,v2.vetor) * 100,2) as sim_100
    from testes.vetores v1, testes.vetores v2 where v1.seq_documento <> v2.seq_documento and dot_product(v1.vetor,v2.vetor)> 0.7

/* tabela para agrupamento de vetores */
CREATE TABLE testes.grupos (
  seq_documento int(11) NOT NULL COMMENT 'seq do documento agrupado' ,
  grupo int(11) NOT NULL DEFAULT -1 COMMENT 'numeração automática do grupo criado pela procedure de agrupamento' ,
  sessao varchar(100) NOT NULL COMMENT 'nome da sessão de agrupamento - o nome é livre e compõe uma lista de vetores para agrupar' ,
  centroide tinyint(1) DEFAULT 0 COMMENT 'será eleito de acordo com a densidade de peças próximas' ,
  sim double DEFAULT NULL COMMENT 'similaridade do documento com o centróide do agrupamento' ,
  dthr timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'data da inclusão do documento no grupo' ,
  PRIMARY KEY (seq_documento, sessao)
) 

/* tabela de log do agrupamento */
drop table testes.grupos_logs;
CREATE TABLE testes.grupos_logs(
   sessao varchar(100), 
   ordem int,
   dthr timestamp default now(), 
   txt_log varchar(100),
   key(sessao, ordem)
   USING CLUSTERED COLUMNSTORE);

/* tabela temporária para inclusão da matriz de comparação entre os vetores durante o agrupamento */
drop table testes.grupos_matriz_tmp;
CREATE TABLE testes.grupos_matriz_tmp(
   sessao varchar(100), 
   seq_documento int, 
   seq_documento_sim int, 
   sim double, 
   centroide BOOL DEFAULT false NOT NULL);
/* no caso de pouca memória, pode-se usar a tabela temporária em disco - um pouco mais lenta*/
CREATE TABLE testes.grupos_matriz_tmp(
   sessao varchar(100), 
   seq_documento int, 
   seq_documento_sim int, 
   sim double, 
   centroide BOOL DEFAULT false NOT NULL,
   KEY (sessao, seq_documento, seq_documento_sim)
   USING CLUSTERED COLUMNSTORE);

