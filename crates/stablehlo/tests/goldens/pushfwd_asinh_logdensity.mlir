module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = chlo.sinh %arg0 : tensor<f32> -> tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.subtract %1, %2 : tensor<f32>
    %7 = stablehlo.divide %6, %0 : tensor<f32>
    %8 = stablehlo.constant dense<-0.5> : tensor<f32>
    %9 = stablehlo.multiply %7, %7 : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.constant dense<"0x8E3F6BBF"> : tensor<f32>
    %12 = stablehlo.add %11, %10 : tensor<f32>
    %13 = stablehlo.abs %1 : tensor<f32>
    %14 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %15 = stablehlo.compare EQ, %13, %14 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.select %15, %16, %12 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.abs %arg0 : tensor<f32>
    %20 = stablehlo.constant dense<"0x000000C0"> : tensor<f32>
    %21 = stablehlo.multiply %20, %18 : tensor<f32>
    %22 = stablehlo.exponential %21 : tensor<f32>
    %23 = stablehlo.log_plus_one %22 : tensor<f32>
    %24 = stablehlo.add %18, %23 : tensor<f32>
    %25 = stablehlo.constant dense<"0x1872313F"> : tensor<f32>
    %26 = stablehlo.subtract %24, %25 : tensor<f32>
    %27 = stablehlo.negate %26 : tensor<f32>
    %28 = stablehlo.subtract %17, %27 : tensor<f32>
    return %28 : tensor<f32>
  }
}
