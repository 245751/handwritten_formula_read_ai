# 手書きで書いてある計算式を読み取るAI

## 概要

このAIは画像に書かれた計算式を読み取り出力するAIです。

## local環境での動作方法

以下のコマンドはgitcloneを行った後に実行してください。

```bash
pip install -r requirements.txt
```

demoを行う場合は仮想環境をjupyterのカーネルに登録が必要なため以下も実行してください。

```bash
pip install ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
```

jupyterでカーネルの選択を求められたときはmyenvを選択してください。




