### pass list
* DShape
* Solovev

### fail list, see ./logs.
* Heliotron: Flux surfaces are no longer nested.
* ESTELL: Flux surfaces are no longer nested.
* WISTELL-A: Flux surfaces are no longer nested.
WISTELL-A and ESTELL are vacuum cases. I know this is a bad test for tokamaks, but I think it's ok for stellarators.

## Small bugs
When using DESC to automatically convert a VMEC input to DESC input,
* DESC forgets to scale current profile coefficients by VMEC's CURTOR scaling factor.
* DESC doesn't check if vmec current profile coefficients are specified in power_series (enclosed current derivative) or power_series_I (enclosed current).
* When symmetry is off in VMEC input, DESC converts this to `sym = False` in the DESC input file. The resulting input file will not run. DESC should instead convert this to `sym = 0`.
* keep seeing "RuntimeWarning: Save attribute '_iota' was not saved as it does not exist." in fix-current mode and vice versa.
