# Local-build shim: Liquid 4.0.3 (pinned by github-pages / Jekyll 3.9) calls
# Object#tainted?, which Ruby 3.2 removed. GitHub Pages ignores _plugins,
# so this only affects local `jekyll serve` on recent Ruby.
class Object
  def tainted?; false; end unless method_defined?(:tainted?)
  def taint; self; end unless method_defined?(:taint)
  def untaint; self; end unless method_defined?(:untaint)
end
