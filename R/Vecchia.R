# Load the module when the package is loaded
.onLoad <- function(libname, pkgname) {
    loadModule("VecchiaGB", TRUE, loadNow = TRUE)
}