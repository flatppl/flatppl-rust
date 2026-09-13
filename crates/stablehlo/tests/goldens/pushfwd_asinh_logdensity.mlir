module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = chlo.sinh %arg0 : tensor<f32> -> tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.subtract %0, %1 : tensor<f32>
    %6 = stablehlo.divide %5, %2 : tensor<f32>
    %7 = stablehlo.constant dense<-0.5> : tensor<f32>
    %8 = stablehlo.multiply %6, %6 : tensor<f32>
    %9 = stablehlo.multiply %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<-0.9189385175704956> : tensor<f32>
    %11 = stablehlo.add %10, %9 : tensor<f32>
    %12 = stablehlo.abs %arg0 : tensor<f32>
    %14 = stablehlo.constant dense<-2.0> : tensor<f32>
    %15 = stablehlo.multiply %14, %12 : tensor<f32>
    %16 = stablehlo.exponential %15 : tensor<f32>
    %17 = stablehlo.log_plus_one %16 : tensor<f32>
    %18 = stablehlo.add %12, %17 : tensor<f32>
    %19 = stablehlo.constant dense<0.6931471824645996> : tensor<f32>
    %20 = stablehlo.subtract %18, %19 : tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.subtract %11, %21 : tensor<f32>
    return %22 : tensor<f32>
  }
}
