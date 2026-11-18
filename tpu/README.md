# Modelos em TPU no Hub

Abra **Cotas e credenciais → Kaggle → Modelos em TPU**. Baixe o notebook, importe-o no Kaggle e selecione **TPU v5e-8** com Internet. Anexe os datasets de pesos indicados na primeira célula. Depois da mensagem de disponibilidade, vincule o endereço e a chave da sessão na aba do Hub. O modelo aparece no seletor do chat da conta que o vinculou. Se o Kaggle reiniciar a sessão, vincule o novo endereço.

## Motor obrigatório da Release

Os notebooks do GLM-5.3-Flash e do Qwen3.8-27B baixam os ZIPs pré-compilados de [Motoria / tpu-v5e8-v1](https://github.com/stefystars-one/motoria/releases/tag/tpu-v5e8-v1) e conferem o SHA-256 antes de extrair. Os ZIPs contêm executáveis do cache XLA/JAX e, no GLM, programas exportados. O notebook ainda carrega os pesos nos oito chips.

O carregamento exige oito chips v5e e um perfil compatível com o pacote. O Qwen confere o Python e as versões de JAX, jaxlib, libtpu, torch, torchax e vLLM-TPU do manifesto. Se o JAX tentar compilar um executável de TPU que não encontrou, a tentativa termina com `MOTORIA_CACHE_MISS`. No GLM, um programa exportado ausente ou incompatível também causa erro. [Cache do JAX](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)

O Qwen3.8-27B Coder usa um [checkpoint separado, fixado por revisão](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated/tree/14c13fcf593d692a5b96f29afecca6fe2fc030e6). Ele permanece indisponível no seletor até o Motoria ter o cache próprio validado e publicado.

## Arquivos locais

A pasta `cache/` guarda os caches baixados dos datasets de referência. A pasta `bundles/` guarda os ZIPs e seus hashes. Para refazer o empacotamento:

```powershell
python tools/kaggle-tpu/download_cache.py glm53-flash qwen38-27b
python tools/kaggle-tpu/pack_bundle.py glm53-flash tools/kaggle-tpu/cache/glm53-flash/v1
python tools/kaggle-tpu/pack_bundle.py qwen38-27b tools/kaggle-tpu/cache/qwen38-27b/v1
python tools/kaggle-tpu/prepare_notebooks.py
```

A origem dos caches está registrada em `hub-engine.json`, dentro de cada ZIP. Os motores de referência vêm do [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab/tree/9a31b8519a80821bcffc5424f6444a9a69bd2711).


## Variante GLM Huihui

O GLM sem censura é a opção recomendada do Hub. O motor pré-compilado é o mesmo da base, com as matrizes de atenção e do especialista compartilhado adaptadas na CPU às camadas 15 a 35 da variante Huihui. Os especialistas roteados permanecem na base original, como no checkpoint publicado pelo autor. A requantização preserva formas e tipos dos argumentos do motor. Ela não é uma cópia bit a bit do GGUF.

O pacote adicional `glm53-flash-abliterated-weights-v1.zip` vem da mesma Release e usa SHA-256 `21c45185209db38b5a0083f65dc7a36097f567735516218e41806ae885347e1e`. A base de cada camada é conferida antes da aplicação. A conversão e a integridade foram verificadas localmente. Qualidade, taxa de recusas e velocidade precisam ser medidas na TPU. A publicação do pacote não comprova 100 tokens/s.

O início automático pelo pool e por contas pessoais autorizadas usa o notebook da variante. O seletor libera a inferência depois de verificar a sessão e o identificador do modelo, mantendo a exigência de motor pronto e recusando compilação de TPU.

Origem dos pesos e intervalo das camadas: [modelo Huihui](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF).
