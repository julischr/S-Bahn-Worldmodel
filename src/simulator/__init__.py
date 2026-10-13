"""Stufe 4: gelernter Simulator der Stammstrecke (siehe outputs/simulator/plan.md).

Hinweis macOS ohne Homebrew: LightGBM braucht libomp. In der .venv wurde die mit
scikit-learn gelieferte libomp nach lightgbm/lib/ kopiert (install_name @rpath) und
@loader_path als rpath in lib_lightgbm.dylib eingetragen (neu signiert).
"""
