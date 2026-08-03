module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = chlo.sinh %arg0 : tensor<f32> -> tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.log %2 : tensor<f32>
    %4 = stablehlo.negate %3 : tensor<f32>
    %5 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %6 = stablehlo.subtract %0, %1 : tensor<f32>
    %7 = stablehlo.divide %6, %2 : tensor<f32>
    %8 = stablehlo.constant dense<-0.5> : tensor<f32>
    %9 = stablehlo.multiply %7, %7 : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.add %4, %5 : tensor<f32>
    %12 = stablehlo.add %11, %10 : tensor<f32>
    %13 = stablehlo.constant dense<-0.5> : tensor<f32>
    %14 = stablehlo.constant dense<1.0> : tensor<f32>
    %15 = chlo.sinh %arg0 : tensor<f32> -> tensor<f32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.power %15, %16 : tensor<f32>
    %18 = stablehlo.add %14, %17 : tensor<f32>
    %19 = stablehlo.log %18 : tensor<f32>
    %20 = stablehlo.multiply %13, %19 : tensor<f32>
    %21 = stablehlo.subtract %12, %20 : tensor<f32>
    return %21 : tensor<f32>
  }
}
