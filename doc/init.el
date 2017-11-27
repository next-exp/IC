(require 'package)
(setq package-enable-at-startup nil)
(add-to-list 'package-archives '("melpa" . "http://melpa.org/packages/"))
(package-initialize)

(unless (package-installed-p 'use-package)
  (package-refresh-contents)
  (package-install 'use-package))

(eval-when-compile
  (require 'use-package))
(require 'bind-key)

(setq use-package-always-ensure t)
(put 'use-package 'lisp-indent-function 1)


(use-package magit

  :bind  (("M-g M-s" . magit-status)
          ("M-g M-d" . magit-dispatch-popup)
          ("C-x g"   . magit-status)
          ("C-x C-g" . magit-dispatch-popup))

  :config
  (setq magit-last-seen-setup-instructions "1.4.0")
  (global-magit-file-mode 1)
  (magit-define-popup-switch 'magit-log-popup ?p "first parent" "--first-parent"))


(use-package helm
  :bind (("C-c h h" . helm-command-prefix)
         ("M-x"     . helm-M-x)
         ("C-x C-f" . helm-find-files)
         ("M-y"     . helm-show-kill-ring)
         ("C-x b"   . ido-switch-buffer)

         :unbind "C-x c"                ; Too similar to C-x C-c !

         :map helm-map
         ;; ("<tab>" . helm-execute-persistent-action) ; Bug: gets
         ;; bound in global map ! so I'm doing it below with
         ;; define-key instead
         ;;("C-i"   . helm-execute-persistent-action)
         ("C-z"   . helm-select-action)

         :map
         helm-command-prefix
         ("g"     . helm-do-grep)
         ("o"     . helm-occur))

  :config
  (require 'helm-config)
  ;; Bug: :bind :map doesn't seem to work properly for special keys,
  ;; and bind <tab> in the global map.
  (bind-key "<tab>" #'helm-execute-persistent-action helm-map)

  (setq helm-split-window-in-side-p           t ; open helm buffer inside current window, not occupy whole other window
        helm-move-to-line-cycle-in-source     t ; move to end or beginning of source when reaching top or bottom of source
        helm-ff-search-library-in-sexp        t ; search for library in `require' and `declare-function' sexp
        helm-scroll-amount                    8 ; scroll 8 lines other window usin M-<next>/M-<prior>
        helm-ff-file-name-history-use-recentf t
        helm-M-x-fuzzy-match                  t
        helm-move-to-line-cycle-in-source     nil
        helm-command-prefix-key               "C-z")

  (helm-mode 1)
  (helm-autoresize-mode t))




(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(indent-tabs-mode nil)
 '(package-selected-packages (quote (helm magit use-package)))
 '(show-paren-mode t))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
)
