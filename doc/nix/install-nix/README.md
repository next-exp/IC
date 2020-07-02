# Installing Nix

Maybe you don't need to install Nix, because it has already been done! Read the
relevant section below, for instructions on how to check.

## Tell me how to install Nix

In many cases it is very easy indeed. In some others its not quite as plain
sailing.

1. If you are on Linux or a pre-Catalina Mac

   ```shell
   sudo curl -L https://nixos.org/nix/install | sh
   ```
2. If you are on a Catalina Mac with a T2 chip

   ```shell
   sh <(curl -L https://nixos.org/nix/install) --darwin-use-unencrypted-nix-store-volume
   ```

3. If you are on Catalina with a pre-T2 chip, please ask for personal help.


In cases 1. and 2., if the process finished without error, you should then

```shell
. $HOME/.nix-profile/etc/profile.d/nix.sh
```
(which line should be added to your `.bash_profile`/`.zshrc`/etc. In some
circumstances the installer does this for you automatically)

Thereafter simply `cd` into the IC source directory and run the IC tests with

```shell
nix-shell    # Installs packages, compiles Cython, prepares environment
ic-test-par  # Runs IC tests in parallel
```

You can hack away on IC to your heart's content in that shell. If you need to
recompile any Cython modules: `ic-compile`.

That's all there is to it!

However, now that we have Nix, we can put it to great use. See
[bootstrap-home-manager](../home-manager/README.md
"bootstrap-home-manager")

## Has Nix already been installed on this machine?
Try
```shell
nix-shell -p cowsay --run 'cowsay Nix is available!'
```
If this eventually leads to the appearance of this joyous message
```
< Nix is available! >
 -------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```
then Nix is already available to this user on this machine.

### That didn't work

Check whether the directory `/nix` exists.

+ If `/nix` is absent, proceed to the installation instructions above
+ If `/nix` is present, check whether either of these files exist:
  - `$HOME/.nix-profile/etc/profile.d/nix.sh`
  - `/etc/profile.d/nix.sh`

  If Neither of these files exist, but `/nix` does, then seek help.

  If one of these exists, it needs to be sourced for Nix to work. In the first
  case this look like:

  ```shell
  source $HOME/.nix-profile/etc/profile.d/nix.sh
  ```

  You should add this line to (or its equivalent for `/etc/profile.d/nix.sh`) to
  your shell initialisation file, and try the `cowsay` test again in a new
  shell.
