# Install Nix

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
(which line should be added to your `.bash_profile`/`.zshrc`/etc.)

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
