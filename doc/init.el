;;; bootstrapping elget
(add-to-list 'load-path "~/.emacs.d/el-get/el-get")

(unless (require 'el-get nil 'noerror)
  (with-current-buffer
      (url-retrieve-synchronously
       "https://raw.githubusercontent.com/dimitri/el-get/master/el-get-install.el")
    (goto-char (point-max))
    (eval-print-last-sexp)))

(add-to-list 'el-get-recipe-path "~/.emacs.d/el-get-user/recipes")
(el-get 'sync)

;; (setq el-get-user-package-directory "~/.emacs.d/el-get/el-get-init-files/")
(setq
 el-get-sources
 '(el-get

   (:name magit
          :after (progn
                   (require 'magit)
                   ;;(setq magit-last-seen-setup-instructions "1.4.0")
                   (global-set-key (kbd "C-x g") 'magit-status)))

   (:name helm
          :after (progn
                   (require 'helm)
                   (require 'helm-config)
                   (define-key helm-map (kbd "<tab>") 'helm-execute-persistent-action)
                   (define-key helm-map (kbd "C-z") 'helm-select-action)

                   (setq helm-split-window-in-side-p           t ; open helm buffer inside current window, not occupy whole other window
                         helm-move-to-line-cycle-in-source     t ; move to end or beginning of source when reaching top or bottom of source
                         helm-ff-search-library-in-sexp        t ; search for library in `require' and `declare-function' sexp
                         helm-scroll-amount                    8 ; scroll 8 lines other window usin M-<next>/M-<prior>
                         helm-ff-file-name-history-use-recentf t)

                   (helm-mode 1)
                   (helm-autoresize-mode t)
                   (setq helm-M-x-fuzzy-match t)
                   (magit-define-popup-switch 'magit-log-popup ?p "first parent" "--first-parent")
                   ;; the default is dangerously close to C-x C-c which kills emacs
                   ;; (global-unset-key (kbd "C-x c")) ; done in custom
                   (global-set-key (kbd "C-x C-f") 'helm-find-files)
                   (global-set-key (kbd "M-x") 'helm-M-x)
                   (global-set-key (kbd "M-y") 'helm-show-kill-ring)
                   (global-set-key (kbd "C-x b") 'ido-switch-buffer)))

   (:name helm-ag
          :after (progn (require 'helm-ag)))

   (:name helm-swoop
          :after (progn
                   (require 'helm-swoop)

                   ;; I feel a hydra on a key-chord coming on. Or maybe just access it via isearch forward
                   (global-set-key (kbd "M-i") 'helm-swoop)
                   (global-set-key (kbd "M-I") 'helm-swoop-back-to-last-point)
                   (global-set-key (kbd "C-c M-i") 'helm-multi-swoop)
                   (global-set-key (kbd "C-x M-i") 'helm-multi-swoop-all)

                   ;; When doing isearch, hand the word over to helm-swoop
                   (define-key isearch-mode-map (kbd "M-i") 'helm-swoop-from-isearch)
                   ;; From helm-swoop to helm-multi-swoop-all
                   (define-key helm-swoop-map (kbd "M-i") 'helm-multi-swoop-all-from-helm-swoop)

                   ;; Move up and down like isearch
                   (define-key helm-swoop-map (kbd "C-r") 'helm-previous-line)
                   (define-key helm-swoop-map (kbd "C-s") 'helm-next-line)
                   (define-key helm-multi-swoop-map (kbd "C-r") 'helm-previous-line)
                   (define-key helm-multi-swoop-map (kbd "C-s") 'helm-next-line)

                   ;; Save buffer when helm-multi-swoop-edit complete
                   (setq helm-multi-swoop-edit-save t)

                   ;; If this value is t, split window inside the current window
                   (setq helm-swoop-split-with-multiple-windows nil)

                   ;; Split direcion. 'split-window-vertically or 'split-window-horizontally
                   (setq helm-swoop-split-direction 'split-window-horizontally)

                   ;; If nil, you can slightly boost invoke speed in exchange for text color
                   (setq helm-swoop-speed-or-color nil)

                   ;; Go to the opposite side of line from the end or beginning of line
                   (setq helm-swoop-move-to-line-cycle t)

                   ;; Optional face for line numbers
                   ;; Face name is `helm-swoop-line-number-face`
                   (setq helm-swoop-use-line-number-face t)))))

(require 'el-get-elpa)
(unless (file-directory-p el-get-recipe-path-elpa)
 (el-get-elpa-build-local-recipes))

;; install new packages and init already installed packages
(el-get 'sync (mapcar 'el-get-source-name el-get-sources))

(require 'package)

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(helm-command-prefix-key "C-z")
 '(indent-tabs-mode nil)
 '(show-paren-mode t))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
)
