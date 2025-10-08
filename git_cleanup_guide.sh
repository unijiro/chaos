#!/bin/bash
# Gitリポジトリ剪定スクリプト
# 巨大ファイルを履歴から完全に削除

echo "================================================"
echo "Git Repository Cleanup Script"
echo "================================================"

# 現在のサイズを表示
echo -e "\n【現在のリポジトリサイズ】"
du -sh .git
git count-objects -vH

echo -e "\n【方法1：最新の状態で新規リポジトリを作成（最も簡単）】"
echo "# 現在の作業ディレクトリをバックアップ"
echo "cd .."
echo "cp -r chaos chaos_backup"
echo ""
echo "# 新しいGitリポジトリを初期化"
echo "cd chaos"
echo "rm -rf .git"
echo "git init"
echo "git add .gitignore *.py *.csv *.ipynb"
echo "git commit -m 'Clean repository with essential files only'"
echo ""
echo "# リモートリポジトリに強制プッシュ（注意：履歴が消えます）"
echo "git remote add origin <your-repo-url>"
echo "git push -u origin main --force"

echo -e "\n【方法2：git filter-branchで特定ファイルを履歴から削除】"
echo "# バックアップを作成"
echo "cd .."
echo "cp -r chaos chaos_backup"
echo "cd chaos"
echo ""
echo "# 巨大ファイルを履歴から削除"
echo "git filter-branch --force --index-filter \\"
echo "  'git rm --cached --ignore-unmatch *.pkl *.png' \\"
echo "  --prune-empty --tag-name-filter cat -- --all"
echo ""
echo "# リモートの参照を削除"
echo "git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin"
echo ""
echo "# ガベージコレクション"
echo "git reflog expire --expire=now --all"
echo "git gc --prune=now --aggressive"
echo ""
echo "# 結果を確認"
echo "du -sh .git"

echo -e "\n【方法3：BFG Repo-Cleaner（最も高速）】"
echo "# BFGをダウンロード"
echo "wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar"
echo ""
echo "# 50MB以上のファイルを削除"
echo "java -jar bfg-1.14.0.jar --strip-blobs-bigger-than 50M ."
echo ""
echo "# ガベージコレクション"
echo "git reflog expire --expire=now --all"
echo "git gc --prune=now --aggressive"

echo -e "\n【方法4：現在のコミットをリセット（履歴が1つだけの場合）】"
echo "# 最初のコミットを修正"
echo "git reset --soft HEAD~1"
echo "git reset"
echo "git add .gitignore *.py *.csv *.ipynb"
echo "git commit -m 'Initial commit with essential files only'"
echo "git push origin main --force"

echo -e "\n================================================"
echo "推奨：方法4（履歴が1つだけなので最も簡単）"
echo "================================================"

cat << 'EOF'

実行手順：

1. バックアップを作成
   cd /home/hayato/study
   cp -r chaos chaos_backup

2. 最初のコミットをリセット
   cd chaos
   git reset --soft HEAD~1
   git reset

3. 必要なファイルのみをコミット
   git add .gitignore
   git add *.py *.csv *.ipynb
   git commit -m "Initial commit with essential files only"

4. リモートに強制プッシュ
   git push origin main --force

5. サイズを確認
   du -sh .git
   git count-objects -vH

これで .git ディレクトリが 477MB → 数MB に削減されます！

EOF
