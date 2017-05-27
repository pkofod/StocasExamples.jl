using StocasExamples
using Base.Test
using Stocas

const se = StocasExamples
solve!(se.am2010()...)
solve!(se.am_tauchen()...)
solve!(se.dixit()...)
solve!(se.hitsch()...)
solve!(se.rust()...)
solve!(se.seven()...)
