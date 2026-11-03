module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = stablehlo.constant dense<"0x00000040"> : tensor<f32>
    %8 = stablehlo.add %7, %arg1 : tensor<f32>
    %9 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %10 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %15 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %16 = stablehlo.add %9, %11 : tensor<f32>
    %17 = stablehlo.add %16, %15 : tensor<f32>
    %18 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %19 = stablehlo.log %18 : tensor<f32>
    %20 = stablehlo.negate %19 : tensor<f32>
    %21 = stablehlo.log %arg0 : tensor<f32>
    %22 = stablehlo.add %21, %20 : tensor<f32>
    %23 = stablehlo.multiply %7, %22 : tensor<f32>
    %24 = stablehlo.log %arg1 : tensor<f32>
    %25 = stablehlo.add %24, %20 : tensor<f32>
    %26 = stablehlo.multiply %arg1, %25 : tensor<f32>
    %27 = stablehlo.add %17, %23 : tensor<f32>
    %28 = stablehlo.add %27, %26 : tensor<f32>
    return %28, %5 : tensor<f32>, tensor<i32>
  }
}
