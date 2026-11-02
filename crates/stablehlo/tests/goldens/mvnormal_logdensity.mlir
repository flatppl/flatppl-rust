module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<[0.2, 0.1]> : tensor<2xf32>
    %1 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %2 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %3 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %4 = stablehlo.compare EQ, %2, %3 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %5 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %6 = stablehlo.select %4, %1, %5 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %9 = stablehlo.constant dense<0.0> : tensor<f32>
    %10 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %12 = stablehlo.compare GT, %8, %11 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %13 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %14 = stablehlo.compare LT, %8, %13 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %15 = stablehlo.select %12, %14, %12 : (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %16 = stablehlo.constant dense<true> : tensor<i1>
    %17 = stablehlo.reduce(%15 init: %16) applies stablehlo.and across dimensions = [0] : (tensor<2xi1>, tensor<i1>) -> tensor<i1>
    %18 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.constant dense<9> : tensor<i32>
    %20 = stablehlo.select %17, %18, %19 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %21 = stablehlo.log %8 : tensor<2xf32>
    %22 = stablehlo.reduce(%21 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.constant dense<2.0> : tensor<f32>
    %24 = stablehlo.multiply %23, %22 : tensor<f32>
    %25 = stablehlo.constant dense<-0.5> : tensor<f32>
    %26 = stablehlo.multiply %25, %24 : tensor<f32>
    %27 = stablehlo.subtract %0, %arg0 : tensor<2xf32>
    %28 = stablehlo.reshape %27 : (tensor<2xf32>) -> tensor<1x2xf32>
    %29 = "stablehlo.triangular_solve"(%1, %28) <{left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>}> : (tensor<2x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %30 = stablehlo.reshape %29 : (tensor<1x2xf32>) -> tensor<2xf32>
    %31 = stablehlo.multiply %30, %30 : tensor<2xf32>
    %32 = stablehlo.reduce(%31 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %33 = stablehlo.multiply %25, %32 : tensor<f32>
    %34 = stablehlo.constant dense<-1.8378770664093453> : tensor<f32>
    %35 = stablehlo.add %34, %26 : tensor<f32>
    %36 = stablehlo.add %35, %33 : tensor<f32>
    return %36, %20 : tensor<f32>, tensor<i32>
  }
}
